"""Fail-closed preemption routing, checkpoint atomicity and single-run history."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from hoi_tables import preemption as recovery


def line(i,value=1):return (json.dumps({'iteration':i,'stage':'actor_critic','wm_total':value})+'\n').encode()
def metrics(root,rank,iterations,value=1):
    p=root/f'rank_{rank:02d}/metrics.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(b''.join(line(i,value) for i in iterations));return p
def checkpoint(root,rank,iteration,payload=b'a'):
    p=root/f'rank_{rank:02d}/checkpoint_{iteration:06d}.pt';p.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p,'w') as f:f.writestr('archive/data.pkl',payload)
    return p


class FakeScheduler:
    def __init__(self):self.states={'100':{'state':'PREEMPTED'},'200':{'state':'RUNNING'}};self.jobs={};self.calls=[];self.released=[]
    def status(self,job):return self.states.get(str(job),{'state':'PENDING'})
    def find(self,name,created):return [j for j,n in self.jobs.items() if n==name]
    def submit(self,args):
        job=str(300+len(self.calls));name=next(s.split('=',1)[1] for s in args if s.startswith('--job-name='))
        self.jobs[job]=name;self.calls.append(args);self.states[job]={'state':'PENDING'};return job
    def stop_logger(self,job):self.states[str(job)]={'state':'COMPLETED'}
    def release(self,job):self.released.append(str(job))


class Tests(unittest.TestCase):
    def setUp(self):
        cache=Path(__file__).resolve().parents[1]/'.cache/preemption_tests';cache.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=cache);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.logical=self.root/'logical';self.actual=self.root/'actual'

    def test_only_preemption_retries(self):
        self.assertTrue(recovery.can_recover('PREEMPTED'))
        for state in ('FAILED','OUT_OF_MEMORY','NODE_FAIL','TIMEOUT','CANCELLED','COMPLETED','UNKNOWN','RUNNING'):
            self.assertFalse(recovery.can_recover(state),state)

    def test_unpaired_tail_preserved_and_replayed_history_never_overwritten(self):
        for rank in range(8):metrics(self.logical,rank,range(501,504))
        p=self.logical/'rank_00/metrics.jsonl';p.write_bytes(p.read_bytes()+line(504)+b'{"iteration":505')
        before=[(self.logical/f'rank_{r:02d}/metrics.jsonl').read_bytes() for r in range(1,8)]
        self.assertEqual(recovery.align_metrics(self.logical,self.root/'archive'),503)
        self.assertEqual((self.root/'archive/rank_00_unpaired_tail.jsonl').read_bytes(),line(504)+b'{"iteration":505')
        for rank in range(8):metrics(self.actual,rank,range(501,507),value=99)
        bridge=recovery.Bridge(self.actual,self.logical);self.assertEqual(bridge.flush(),506)
        for rank in range(8):
            raw=(self.logical/f'rank_{rank:02d}/metrics.jsonl').read_bytes()
            self.assertEqual(raw,b''.join(line(i) for i in range(501,504))+b''.join(line(i,99) for i in range(504,507)))
        self.assertEqual(bridge.flush(),506)
        self.assertEqual(recovery.Bridge(self.actual,self.logical).flush(),506)

    def test_partial_json_and_partial_rank_are_not_published(self):
        for rank in range(8):metrics(self.logical,rank,[501]);metrics(self.actual,rank,[502])
        p=self.actual/'rank_07/metrics.jsonl';p.write_bytes(line(502)[:-1])
        bridge=recovery.Bridge(self.actual,self.logical);self.assertEqual(bridge.flush(),501)
        with p.open('ab') as f:f.write(b'\n')
        self.assertEqual(bridge.flush(),502)

    def test_divergent_rank_iterations_fail_before_append(self):
        for rank in range(8):metrics(self.logical,rank,[501]);metrics(self.actual,rank,[502 if rank!=7 else 503])
        with self.assertRaisesRegex(ValueError,'diverged'):recovery.Bridge(self.actual,self.logical).flush()

    def test_earlier_prefix_corruption_is_not_silently_trimmed(self):
        for rank in range(8):metrics(self.logical,rank,[501,502] if rank!=7 else [502])
        with self.assertRaisesRegex(ValueError,'earlier common prefix'):recovery.align_metrics(self.logical,self.root/'archive')

    @unittest.skipIf(os.name=='nt','Production checkpoint links are verified on Linux')
    def test_checkpoint_published_only_when_all_eight_files_complete(self):
        for rank in range(8):metrics(self.logical,rank,[501]);metrics(self.actual,rank,[502])
        bridge=recovery.Bridge(self.actual,self.logical)
        for rank in range(7):checkpoint(self.actual,rank,1000)
        broken=self.actual/'rank_07/checkpoint_001000.pt';broken.write_bytes(b'partial zip')
        bridge.flush();self.assertFalse((self.logical/'rank_00/checkpoint_001000.pt').exists())
        checkpoint(self.actual,7,1000);bridge.flush()
        for rank in range(8):self.assertEqual((self.logical/f'rank_{rank:02d}/checkpoint_001000.pt').resolve(),(self.actual/f'rank_{rank:02d}/checkpoint_001000.pt').resolve())

    def setup_handler(self):
        campaign=self.root/'campaign';campaign.mkdir();recovery.atomic(campaign/'experiment.json',{'paths':{}})
        controller=self.root/'controller';controller.mkdir()
        policy={'campaign':str(campaign),'training_root':str(self.logical),'run_id':'owner001','source':str(self.root/'source'),'helpers':str(self.root/'helpers')}
        for rank in range(8):metrics(self.logical,rank,[501,502])
        recovery.atomic(controller/'state.json',{'current':{'attempt':0,'train_job':'100','logger_job':'200','actual_training_root':str(self.logical)},'handled':{},'handlers':{}})
        return controller,policy

    @unittest.skipIf(os.name=='nt','Production controller uses Linux flock')
    def test_handler_held_submission_is_idempotent_and_arms_next_dependency(self):
        controller,policy=self.setup_handler();scheduler=FakeScheduler()
        with patch.object(recovery,'settings',return_value=policy),patch.object(recovery,'audit_bundle',return_value=(str(self.root/'fallback.json'),500)):
            result=recovery.handle(controller,'100',scheduler)
            self.assertEqual(result['resume_iteration'],500)
            self.assertEqual(len(scheduler.calls),2)
            self.assertIn('--hold',scheduler.calls[0])
            self.assertIn('--dependency=afterany:'+result['job_id'],scheduler.calls[1])
            a=recovery.read(controller/'attempt_001/attempt.json');self.assertEqual(a['run_id'],'owner001')
            self.assertEqual(a['train_job'],result['job_id'])
            self.assertEqual(recovery.handle(controller,'100',scheduler),result)
            self.assertEqual(len(scheduler.calls),2)
            self.assertEqual(recovery.read(controller/'state.json')['current']['logger_job'],None)

    @unittest.skipIf(os.name=='nt','Production controller uses Linux flock')
    def test_cancel_error_completion_and_bad_checkpoint_never_queue_gpu(self):
        controller,policy=self.setup_handler()
        for state in ('FAILED','CANCELLED','COMPLETED','OUT_OF_MEMORY','TIMEOUT'):
            recovery.atomic(controller/'state.json',{'current':{'train_job':'100'},'handled':{}})
            scheduler=FakeScheduler();scheduler.states['100']={'state':state}
            with patch.object(recovery,'settings',return_value=policy):
                self.assertEqual(recovery.handle(controller,'100',scheduler)['reason'],state)
            self.assertFalse(scheduler.calls)

    @unittest.skipIf(os.name=='nt','Production controller uses Linux flock')
    def test_incomplete_future_checkpoint_is_archived_not_deleted(self):
        controller,policy=self.setup_handler();p=checkpoint(self.logical,0,1000);raw=p.read_bytes()
        with patch.object(recovery,'settings',return_value=policy),patch.object(recovery,'audit_bundle',return_value=(str(self.root/'fallback.json'),500)):
            recovery.handle(controller,'100',FakeScheduler())
        self.assertFalse(p.exists());self.assertEqual((controller/'attempt_001/orphan_checkpoints/rank_00/checkpoint_001000.pt').read_bytes(),raw)

    def test_lost_submission_reply_is_reconciled_not_duplicated(self):
        scheduler=FakeScheduler();scheduler.jobs['123']='unique-name'
        record={'name':'unique-name','created_utc':recovery.utc(),'submission_started':recovery.utc()}
        self.assertEqual(recovery.submit_once(record,scheduler,[]),'123');self.assertFalse(scheduler.calls)
        record={'name':'missing','created_utc':recovery.utc(),'submission_started':recovery.utc()}
        with self.assertRaisesRegex(RuntimeError,'refuse duplicate'):recovery.submit_once(record,scheduler,[])

    def test_tables_cannot_be_armed(self):
        policy=self.root/'policy.json';recovery.atomic(policy,{'schema':recovery.SCHEMA,'profile':'tables','assets':list(recovery.ASSETS)})
        with self.assertRaisesRegex(ValueError,'Ultra-only'):recovery.settings(self.root)

    @unittest.skipIf(os.name=='nt','Production controller uses Linux flock')
    def test_bad_resume_bundle_cannot_queue_a_gpu_retry(self):
        controller,policy=self.setup_handler();scheduler=FakeScheduler()
        with patch.object(recovery,'settings',return_value=policy),patch.object(recovery,'audit_bundle',side_effect=ValueError('nonfinite learner state')):
            with self.assertRaisesRegex(ValueError,'nonfinite'):recovery.handle(controller,'100',scheduler)
        self.assertEqual(scheduler.calls,[])

    def test_completed_and_inflight_evaluations_survive_recovery(self):
        entries={'2000':{'status':'complete','result_sha256':'a'*64},'4000':{'status':'submitted','job_id':'42'}}
        state={'finished':True,'parent_training_exitcode':143,'missing_milestones':[6000],'finished_utc':'earlier','evaluations':deepcopy(entries),'source_sha256':'same'}
        recovery.reopen_evaluation_ledger(state)
        self.assertNotIn('finished',state);self.assertEqual(state['evaluations'],entries)
        self.assertEqual(state['source_sha256'],'same')

    def test_logger_resumes_only_the_existing_owner_and_same_eval_namespace(self):
        controller,policy=self.setup_handler();policy.update(source='science',eval_control='original-evals')
        attempt=self.root/'attempt';attempt.mkdir()
        recovery.atomic(attempt/'attempt.json',{'attempt':1,'resume_iteration':1000,'previous_train_job':'100',
            'train_job':'300','metrics_frontier':1100,'control':'new-logging'})
        calls=[];run=SimpleNamespace(summary={})
        module=SimpleNamespace(init=lambda **kw:(calls.append(kw) or run))
        def monitor(*args,**kw):
            self.assertEqual(kw['expected_run_id'],'owner001');self.assertEqual(kw['eval_control'],'original-evals')
            sdk=kw['wandb_module'];sdk.init(id='owner001',resume='must')
            self.assertEqual(run.summary['recovery/checkpoint_iteration'],1000)
            with self.assertRaisesRegex(ValueError,'another run'):sdk.init(id='wrong',resume='never')
            return 0
        with patch.object(recovery,'settings',return_value=policy),patch('hoi_tables.telemetry._sdk',return_value=module),patch('hoi_tables.unified_monitor.monitor',side_effect=monitor):
            self.assertEqual(recovery.logger(controller,attempt),0)
        self.assertEqual(calls,[{'id':'owner001','resume':'must'}])

    def test_cpu_visibility_mask_is_not_inherited_by_new_gpu_submission(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':''}),patch('subprocess.run',return_value=SimpleNamespace(stdout='123\n')) as call:
            self.assertEqual(recovery.Scheduler().submit(['a-script']),'123')
            self.assertNotIn('CUDA_VISIBLE_DEVICES',call.call_args.kwargs['env'])
            self.assertNotIn('ROCR_VISIBLE_DEVICES',call.call_args.kwargs['env'])

    def test_logger_handoff_signals_only_parent_not_sdk_children(self):
        scheduler=recovery.Scheduler()
        with patch.object(scheduler,'status',return_value={'state':'RUNNING'}),patch.object(scheduler,'call') as call:
            scheduler.stop_logger('200')
        self.assertEqual(call.call_args.args[0],['scancel','--signal=TERM','--batch','200'])

    def test_counterfeit_checkpoint_set_is_not_considered_complete(self):
        for rank in range(8):
            p=self.actual/f'rank_{rank:02d}/checkpoint_001000.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'not a complete torch archive')
        self.assertEqual(recovery.complete_sets(self.actual),{})
