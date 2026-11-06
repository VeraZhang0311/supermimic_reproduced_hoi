"""Object DR only; robot DR stays entirely in the unchanged G1PhysicsRuntime."""
import torch

CONTRACT = {"enabled_in_training": True, "object_mass_scale": [.7, 1.3],
            "object_com_delta_m": [-.01, .01], "object_friction": [.4, .8],
            "inertia": "nominal COM tensor multiplied by mass scale",
            "resample": "episode reset", "robot": "inherited privileged GMT hw3",
            "evaluation": "nominal properties, no randomization"}


class ObjectRandomization:
    def __init__(self, obj, sim, profile, num_envs, device, enabled=True):
        self.obj, self.sim, self.enabled = obj, sim, enabled
        self.N, self.device, self.profile = num_envs, device, profile
        self.mass = torch.full((num_envs,), float(profile['mass']), device=device)
        self.com = torch.tensor(profile['com'], device=device).float().expand(num_envs, -1).clone()
        self.inertia = torch.tensor(profile['inertia'], device=device).float().reshape(1,3,3).expand(num_envs,-1,-1).clone()
        self.friction = torch.full((num_envs,), float(profile['friction']), device=device)
        self.nominal = (self.mass.clone(), self.com.clone(), self.inertia.clone(), self.friction.clone())
        self.readback_count = 0

    def reset(self, ids):
        import warp as wp
        from newton import ModelFlags
        if not len(ids):
            return
        m, c, I, mu = (x[ids] for x in self.nominal)
        if self.enabled:
            scale = torch.empty_like(m).uniform_(.7, 1.3)
            m = m * scale
            I = I * scale[:, None, None]
            c = c + torch.empty_like(c).uniform_(-.01, .01)
            mu = torch.empty_like(mu).uniform_(.4, .8)
        for target, sample in zip((self.mass, self.com, self.inertia, self.friction), (m,c,I,mu)):
            target[ids] = sample
        obj, manager = self.obj, self.sim.backend_manager
        obj.set_masses_index(masses=self.mass[:,None].contiguous())
        obj.set_coms_index(coms=self.com[:,None].contiguous())
        obj.set_inertias_index(inertias=self.inertia.reshape(self.N,1,9).contiguous())
        model = manager.get_model()
        shape = wp.to_torch(obj._root_view.get_attribute('shape_material_mu', model))
        if shape.shape[0] != self.N or not shape.numel():
            raise RuntimeError('Object friction view does not preserve world axis')
        mu = self.friction.reshape(self.N, *([1]*(shape.ndim-1))).expand_as(shape).contiguous()
        obj._root_view.set_attribute('shape_material_mu', model, wp.from_torch(mu))
        manager.add_model_change(ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.SHAPE_PROPERTIES)
        checks = [(obj.data.body_mass.torch[:,0], self.mass),
                  (obj.data.body_com_pose_b.torch[:,0,:3], self.com),
                  (obj.data.body_inertia.torch.reshape(self.N,3,3), self.inertia),
                  (wp.to_torch(obj._root_view.get_attribute('shape_material_mu', model)), mu)]
        if any(not torch.allclose(a,b,rtol=1e-5,atol=1e-6) for a,b in checks):
            raise RuntimeError('Object DR native readback mismatch')
        self.readback_count += 1

    def context(self, ground_friction):
        I = self.inertia
        entries = torch.stack((I[:,0,0],I[:,1,1],I[:,2,2],I[:,0,1],I[:,0,2],I[:,1,2]), -1)
        return torch.cat((self.mass[:,None],self.com,entries,self.friction[:,None],
                          torch.full_like(self.mass[:,None], ground_friction)), -1)
