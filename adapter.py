"""ScaleBFM deployment reference-forcing mode expressed against SDK packets."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController


class ScaleBFM(ONNXController):
    def frame(self,state):
        q,qd,w,g,r=self.proprio(state)
        return [np.asarray(state.root_orientation_xyzw)[[3,0,1,2]],w,q,qd,self.last]

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        self.histories=[np.tile(v,(3,1)).astype(np.float32) for v in self.frame(state)]
        self.body_names=self.cfg['selected_body_names']

    def observations(self,state,reference):
        self.histories=[np.concatenate([h[1:],np.asarray(v,np.float32)[None]]) for h,v in zip(self.histories,self.frame(state))]
        bodies=[{b.name:b for b in f.bodies} for f in reference]
        p=np.array([[row[n].position_world_m for n in self.body_names] for row in bodies])
        q=np.array([[row[n].orientation_xyzw for n in self.body_names] for row in bodies])
        actual=R.from_quat(state.root_orientation_xyzw)
        # All canonical episodes start at reference pose, so initial heading alignment is identity.
        # Reference forcing subtracts current reference pelvis, not actual global position.
        relp=actual.inv().apply((p-p[0,0]).reshape(-1,3)).reshape(p.shape)
        relq=(actual.inv()*R.from_quat(q.reshape(-1,4))).as_quat().reshape(q.shape)[...,[3,0,1,2]]
        keys=['root_quat_buffer','base_ang_vel_buffer','dof_pos_buffer','dof_vel_buffer','actions_buffer']
        result={k:h[None] for k,h in zip(keys,self.histories)}
        result.update(target_body_pos_future_to_robot_base=relp[None],target_body_rot_future_to_robot_base=relq[None],
                      control_mode=np.array([7],np.int64),future_time_offsets=np.arange(6,dtype=np.int64)[None,:,None])
        return result


def create(root,runtime):return ScaleBFM(root,runtime)
