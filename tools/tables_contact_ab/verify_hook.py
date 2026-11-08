"""Real pinned-backend contacts: A no-op, B pair isolation and graph replay."""
import json
import os
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from normal_contact import FootTableNormalProbe
from supermimic.g1_contactmix import install_hand_contact_multiply


def check(device):
    with wp.ScopedDevice(device):
        xml='''<mujoco><option gravity="0 0 0" timestep=".005" cone="elliptic"/>
        <worldbody><geom name="ground" type="plane" size="10 10 .1" friction=".3 0 0"/>
        <body name="foot" pos="0 0 1"><geom name="foot" size=".1"/></body>
        <body name="table" pos=".15 0 1"><freejoint/><geom name="table" size=".1" mass="1" friction=".9 0 0"/></body>
        <body name="left" pos="1 0 1"><geom name="left" size=".1"/></body>
        <body name="right" pos="2 0 1"><geom name="right" size=".1"/></body>
        <body name="hand" pos="0 1 1"><geom name="hand" size=".1" priority="1" friction="1 0 0" solref="-18 -8" solimp=".9 .9 .001 .5 2"/></body>
        <body pos=".15 1 1"><freejoint/><geom name="hand_object" size=".1" mass="1" friction=".2 0 0"/></body>
        <body pos="2 0 .05"><freejoint/><geom name="unrelated_ground" size=".1" mass="1"/></body>
        </worldbody></mujoco>'''
        cpu=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(cpu)
        model=mjw.put_model(cpu); d=mjw.put_data(cpu,data,nworld=1,nconmax=64,njmax=200)
        ids={cpu.geom(i).name:i for i in range(cpu.ngeom)}
        roles=np.full(cpu.ngeom,-1,np.int32)
        roles[ids['table']]=2;roles[ids['foot']]=3;roles[ids['right']]=4
        hand=install_hand_contact_multiply(model,[ids['hand']],ground_geom_ids=[ids['ground']])
        def snapshot():
            n=int(d.nacon.numpy()[0]);assert n>=3
            pairs=d.contact.geom.numpy()[:n]
            fields={k:getattr(d.contact,k).numpy()[:n] for k in ('solref','solimp','friction','dim')}
            return {tuple(sorted(map(int,p))):{k:v[i].copy() for k,v in fields.items()} for i,p in enumerate(pairs)}
        mjw.forward(model,d); baseline=snapshot()
        target=tuple(sorted((ids['foot'],ids['table'])))
        assert target in baseline
        geom_before={k:getattr(model,k).numpy().copy() for k in ('geom_solref','geom_solimp','geom_friction')}
        A=FootTableNormalProbe(hand,roles,d.naconmax)
        mjw.forward(model,d); observed=snapshot()
        for key in baseline:
            for field in baseline[key]:np.testing.assert_array_equal(observed[key][field],baseline[key][field])
        A.close();B=FootTableNormalProbe(hand,roles,d.naconmax,.01)
        def verify():
            observed=snapshot();assert observed.keys()==baseline.keys()
            for key in baseline:
                for field in baseline[key]:
                    expected=baseline[key][field].copy()
                    if key==target and field=='solref':expected[0]=.01
                    np.testing.assert_array_equal(observed[key][field],expected)
            for name,expected in geom_before.items():np.testing.assert_array_equal(getattr(model,name).numpy(),expected)
            hand.ensure_installed();B.ensure_installed()
            assert np.isfinite(d.qfrc_constraint.numpy()).all()
        mjw.forward(model,d);verify()
        if device.startswith('cuda'):
            if not wp.is_conditional_graph_supported():model.opt.graph_conditional=False
            with wp.ScopedCapture(device=device) as capture:mjw.forward(model,d)
            wp.capture_launch(capture.graph);verify()
        B.close();hand.ensure_installed();hand.close()
        return {'device':device,'A_bitwise_noop':True,'B_only_target_time_constant':True,
                'unrelated_contacts_hand_friction_and_model_geoms_unchanged':True,
                'cuda_graph_replay':device.startswith('cuda')}


if __name__=='__main__':
    wp.init()
    result=[check('cpu'),check(os.environ.get('AB_GPU_DEVICE','cuda:0'))]
    print(json.dumps({'passed':True,'checks':result}),flush=True)
