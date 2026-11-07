"""Real dynamic ULTRA object alongside the unchanged privileged GMT robot.

One fixed object asset per simulator worker. Multi-asset PPO synchronizes one
actor/critic across workers, so MJWarp worlds remain topologically identical.
No automatic reset: the collector values the final observation before reset.
"""
from pathlib import Path
from types import FunctionType
import torch
from supermimic.env import TrackingSceneCfg, TrackingSim
from supermimic.rotations import mat_to_quat, quat_to_mat
from .assets import prepare_object_usd, sample_surface_points
from .randomization import ObjectRandomization
from .rewards import MAX_POSITIVE_REWARD, reward


class HOISim(TrackingSim):
    def __init__(self, sim_cfg, num_envs, motion, mdp, asset, output, *, eval_mode=False,
                 max_episode_steps=500, v11_reward=None):
        if not eval_mode and (not motion.for_training or motion.partition != 'train'):
            raise ValueError('Physical training refuses Protected eval reference libraries')
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.utils import configclass
        self.asset = asset
        self.asset_index = int(asset['asset_id'])
        self.compatible = motion.compatible_bins(self.asset_index)
        if not len(self.compatible):
            raise ValueError('No train motions for this fixed worker asset')
        self._object_ready = False
        self.v11_reward = v11_reward
        self.object_solver_report = None
        self.object_solver_checks = 0
        usd = prepare_object_usd(asset, Path(output) / 'object_asset')

        @configclass
        class Scene(TrackingSceneCfg):
            object = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Object',
                spawn=sim_utils.UsdFileCfg(usd_path=str(usd), rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0., angular_damping=0.)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(4.,4.,2.)))

        # Execute the exact inherited initializer with a private scene binding.
        # Its bytecode, robot config and global module remain unchanged. This
        # is needed because the frozen GMT constructor has no scene hook and
        # its implementation SHA is part of the checkpoint physics contract.
        original = TrackingSim.__init__
        initialize = FunctionType(original.__code__,
                                  {**original.__globals__, 'TrackingSceneCfg': Scene},
                                  original.__name__, original.__defaults__, original.__closure__)
        initialize(self, sim_cfg, num_envs, motion, mdp, eval_mode=eval_mode,
                   max_episode_steps=max_episode_steps)
        self.object = self.scene['object']
        self.asset_ids = torch.full((self.N,), self.asset_index, dtype=torch.long, device=self.device)
        self.points = sample_surface_points(asset, num_points=1024, seed=2024).to(self.device)
        self.bounds = torch.cat((self.points.amin(0), self.points.amax(0)))
        self.randomizer = ObjectRandomization(self.object, self.sim, asset['physical_profile'],
                                              self.N, self.device, enabled=not eval_mode)
        self._object_ready = True
        self.reset(torch.arange(self.N, device=self.device), self.f.clone())
        max_step_reward = MAX_POSITIVE_REWARD
        if v11_reward is not None:
            from .v11_contacts import HandObjectContacts
            from .v11_reward import MAX_STEP_REWARD
            self.hand_contacts = HandObjectContacts(self)
            max_step_reward = MAX_STEP_REWARD
        self.G_max = max_step_reward / (1 - self.gamma)
        self.verify_dynamic_object()

    def verify_dynamic_object(self):
        import mujoco
        model = self.sim.backend_manager._solver.mj_model
        bodies = [i for i in range(model.nbody) if 'Object' in (model.body(i).name or '')
                  and int(model.body_jntnum[i]) > 0]
        if not bodies or any(model.body_mass[i] <= 0 or int(model.body_jntnum[i]) != 1 or
                             model.jnt_type[model.body_jntadr[i]] != mujoco.mjtJoint.mjJNT_FREE for i in bodies):
            raise RuntimeError('ULTRA object must be a positive-mass dynamic free rigid body')
        self.dynamic_object_evidence = {'bodies': [model.body(i).name for i in bodies], 'passed': True}

    def sample_frames(self, n):
        bins = self.compatible
        if self.eval_mode:
            selected = bins[:1].expand(n)
            return self.motion.clip_start[self.motion.bin_clip[selected]], selected
        soft = torch.softmax(-25 * self.bin_G[bins] / self.G_max, 0)
        self.bin_w_avg[bins] = .999 * self.bin_w_avg[bins] + .001 * soft
        avg = self.bin_w_avg[bins]
        probs = .5 / len(bins) + .4 * soft + .1 * avg / avg.sum()
        selected = bins[torch.multinomial(probs, n, replacement=True)]
        start = self.motion.bin_start[selected]
        end = torch.minimum(start+50, self.motion.clip_end[self.motion.bin_clip[selected]])
        frames = start + (torch.rand(n, device=self.device)*(end-start)).long()
        return frames, selected

    def reset(self, env_ids, frames=None):
        super().reset(env_ids, frames)
        if not self._object_ready or not len(env_ids):
            return
        if not bool((self.motion.clip_asset_ids[self.motion.clip_id[self.f[env_ids]]] == self.asset_index).all()):
            raise RuntimeError('Reference asset does not match the physical world')
        self.randomizer.reset(env_ids)
        self._check_object_after_step = True
        p,R,v,w = self.motion.object_ref(self.f[env_ids])
        quat = mat_to_quat(R)
        pose = torch.cat((p+self.scene.env_origins[env_ids],quat[:,1:],quat[:,:1]), -1)
        self.object.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim_index(root_velocity=torch.cat((v,w),-1), env_ids=env_ids)
        self.refresh()

    def object_state(self):
        data = self.object.data
        pose, vel = data.root_link_pose_w.torch.clone(), data.root_link_vel_w.torch.clone()
        R = quat_to_mat(torch.cat((pose[:,6:7],pose[:,3:6]), -1))
        return pose[:,:3]-self.scene.env_origins, R, vel[:,:3], vel[:,3:]

    def object_context(self):
        return self.randomizer.context(1. if self.eval_mode else .3)

    def step(self, action):
        if not torch.isfinite(action).all():
            raise FloatingPointError('Nonfinite combined GMT/residual action')
        S0, previous = self.state(), self.a_prev.clone()
        target = self.default_q + self.xi[:,1:30] + self.alpha * action
        if self.v11_reward is None:
            self.physics.step_target(target)
        else:
            contact, contact_valid = self.hand_contacts.step(target)
        if self._check_object_after_step:
            from .native_checks import audit_object_solver
            self.object_solver_report = audit_object_solver(self)
            self.object_solver_checks += 1
            self._check_object_after_step = False
        S1, O1, f1 = self.state(), self.object_state(), self.f+1
        if any(not torch.isfinite(x).all() for x in (*S1,*O1)):
            raise FloatingPointError('Nonfinite physical HOI state')
        if self.v11_reward is None:
            r, terms = reward(self.mdp,S1,O1,f1,action,previous,self.points,S0[5])
        else:
            from .rewards import object_distance
            r, terms = self.v11_reward(self.mdp, S1, O1, f1, action, previous, contact, contact_valid)
            # Keep the baseline failure criterion for the reward-only comparison.
            terms['object_surface_error_m'] = object_distance(O1, self.motion.object_ref(f1), self.points)
        term = self.mdp.termination(S1,f1) | (terms['object_surface_error_m'] > .3)
        trunc = (f1 >= self.motion.clip_end[self.motion.clip_id[self.f]]) | (self.ep_step+1 >= self.max_episode_steps)
        if not torch.isfinite(r).all():
            raise FloatingPointError('Nonfinite HOI reward')
        self.ep_disc_ret += self.gamma ** self.ep_step.float() * r
        self.ep_ret += r
        self.f, self.a_prev = f1, action.clone()
        self.ep_step += 1
        done = term | trunc
        ids = done.nonzero().flatten()
        if len(ids) and not self.eval_mode:
            T = self.ep_step[ids].float()
            bootstrap = torch.where(term[ids],0.,self.gamma**T*self.ep_ret[ids]/T/(1-self.gamma))
            bins = self.start_bin[ids]
            # Average duplicate bins instead of a nondeterministic overwrite.
            sums, counts = torch.zeros_like(self.bin_G), torch.zeros_like(self.bin_G)
            sums.scatter_add_(0,bins,self.ep_disc_ret[ids]+bootstrap)
            counts.scatter_add_(0,bins,torch.ones_like(T))
            used = counts > 0
            self.bin_G[used] = .8*self.bin_G[used]+.2*sums[used]/counts[used]
        return {'reward':r,'terminated':term,'truncated':trunc,'done':done,'terms':terms,
                'previous_q':S0[4], 'previous_qd':S0[5]}
