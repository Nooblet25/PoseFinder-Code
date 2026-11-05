using System.Collections.Generic;
using UnityEngine;

namespace PoseFinder
{
    [RequireComponent(typeof(Animator))]
    public sealed class AvatarSkeletonCapture : MonoBehaviour
    {
        private static readonly (string Joint, HumanBodyBones Bone, HumanBodyBones? Fallback)[] Map =
        {
            ("mixamorigHips",         HumanBodyBones.Hips,          null),
            ("mixamorigSpine",        HumanBodyBones.Spine,         null),
            ("mixamorigSpine1",       HumanBodyBones.Chest,         null),
            ("mixamorigSpine2",       HumanBodyBones.UpperChest,    HumanBodyBones.Chest),
            ("mixamorigHead",         HumanBodyBones.Head,          null),
            ("mixamorigLeftArm",      HumanBodyBones.LeftUpperArm,  null),
            ("mixamorigLeftForeArm",  HumanBodyBones.LeftLowerArm,  null),
            ("mixamorigLeftHand",     HumanBodyBones.LeftHand,      null),
            ("mixamorigRightArm",     HumanBodyBones.RightUpperArm, null),
            ("mixamorigRightForeArm", HumanBodyBones.RightLowerArm, null),
            ("mixamorigRightHand",    HumanBodyBones.RightHand,     null),
            ("mixamorigLeftUpLeg",    HumanBodyBones.LeftUpperLeg,  null),
            ("mixamorigLeftLeg",      HumanBodyBones.LeftLowerLeg,  null),
            ("mixamorigLeftFoot",     HumanBodyBones.LeftFoot,      null),
            ("mixamorigRightUpLeg",   HumanBodyBones.RightUpperLeg, null),
            ("mixamorigRightLeg",     HumanBodyBones.RightLowerLeg, null),
            ("mixamorigRightFoot",    HumanBodyBones.RightFoot,     null),
        };

        public Dictionary<string, JointPos> Capture()
        {
            var animator = GetComponent<Animator>();
            if (animator == null || !animator.isHuman)
            {
                Debug.LogError("AvatarSkeletonCapture needs an Animator with a Humanoid avatar.", this);
                return null;
            }

            Quaternion inverseRotation = Quaternion.Inverse(transform.rotation);
            var skeleton = new Dictionary<string, JointPos>(Map.Length);

            foreach (var (joint, bone, fallback) in Map)
            {
                Transform t = animator.GetBoneTransform(bone);
                if (t == null && fallback.HasValue) t = animator.GetBoneTransform(fallback.Value);
                if (t == null) continue;

                Vector3 p = inverseRotation * (t.position - transform.position);
                skeleton[joint] = new JointPos(p.x, p.y, p.z);
            }

            return skeleton;
        }
    }
}
