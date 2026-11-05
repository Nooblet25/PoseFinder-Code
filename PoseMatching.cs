using System;
using System.Collections.Generic;
using System.Linq;

namespace PoseFinder
{
    [Serializable]
    public struct JointPos
    {
        public float x, y, z;
        public JointPos(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
    }

    public readonly struct Pt
    {
        public readonly double X, Y;
        public Pt(double x, double y) { X = x; Y = y; }
        public static Pt operator +(Pt a, Pt b) => new Pt(a.X + b.X, a.Y + b.Y);
        public static Pt operator /(Pt a, double d) => new Pt(a.X / d, a.Y / d);
        public double DistanceTo(Pt o)
        {
            double dx = X - o.X, dy = Y - o.Y;
            return Math.Sqrt(dx * dx + dy * dy);
        }
    }

    public sealed class AvatarAttributes
    {
        public bool IsHorizontal, IsSitting, IsWideStance, LHandUp, RHandUp;
        public double DistLHead, DistRHead, FeetDist, HandSpan, HipsY, ShoulderW, LHandY, RHandY;
    }

    public sealed class PersonDetection
    {
        public Pt[] Keypoints;
        public double[] Confidence;
    }

    public static class Coco
    {
        public const int Nose = 0, LEye = 1, REye = 2, LEar = 3, REar = 4;
        public const int LShoulder = 5, RShoulder = 6, LElbow = 7, RElbow = 8, LWrist = 9, RWrist = 10;
        public const int LHip = 11, RHip = 12, LKnee = 13, RKnee = 14, LAnkle = 15, RAnkle = 16;
    }

    public static class Rig
    {
        public const int Hips = 0, Spine = 1, Spine1 = 2, Shoulders = 3, Head = 4;
        public const int LShoulder = 5, LElbow = 6, LWrist = 7;
        public const int RShoulder = 8, RElbow = 9, RWrist = 10;
        public const int LHip = 11, LKnee = 12, LAnkle = 13;
        public const int RHip = 14, RKnee = 15, RAnkle = 16;
    }

    public static class PoseMatching
    {
        public const double ConfThreshold = 0.35;
        public const int MaxResults = 12;
        public const double MinPersonDetectionConf = 0.3;
        public const double RawMinMatchScore = 50;
        public const double StretchExponent = 0.65;

        public static readonly string[] JointOrder =
        {
            "mixamorigHips", "mixamorigSpine", "mixamorigSpine1", "mixamorigSpine2", "mixamorigHead",
            "mixamorigLeftArm", "mixamorigLeftForeArm", "mixamorigLeftHand",
            "mixamorigRightArm", "mixamorigRightForeArm", "mixamorigRightHand",
            "mixamorigLeftUpLeg", "mixamorigLeftLeg", "mixamorigLeftFoot",
            "mixamorigRightUpLeg", "mixamorigRightLeg", "mixamorigRightFoot",
        };

        private static readonly double[] JointWeights =
            { 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

        private static readonly string[] MinorAgeMarkers = { "child", "kid", "teen" };
        private static readonly string[] MinimalAttireMarkers = { "minimal", "swimwear", "bikini", "underwear" };

        public static double Procrustes(Pt[] a, Pt[] b)
        {
            int n = a.Length;
            if (n == 0 || n != b.Length || n != JointWeights.Length)
                throw new ArgumentException("Poses must have the same length as the joint weights (17).");

            double amx = 0, amy = 0, bmx = 0, bmy = 0;
            for (int i = 0; i < n; i++) { amx += a[i].X; amy += a[i].Y; bmx += b[i].X; bmy += b[i].Y; }
            amx /= n; amy /= n; bmx /= n; bmy /= n;

            var ac = new Pt[n];
            var bc = new Pt[n];
            double na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                ac[i] = new Pt(a[i].X - amx, a[i].Y - amy);
                bc[i] = new Pt(b[i].X - bmx, b[i].Y - bmy);
                na += ac[i].X * ac[i].X + ac[i].Y * ac[i].Y;
                nb += bc[i].X * bc[i].X + bc[i].Y * bc[i].Y;
            }
            na = Math.Sqrt(na);
            nb = Math.Sqrt(nb);
            if (na == 0 || nb == 0) return 0.0;

            var an = new Pt[n];
            var bn = new Pt[n];
            double m00 = 0, m01 = 0, m10 = 0, m11 = 0;
            for (int i = 0; i < n; i++)
            {
                an[i] = new Pt(ac[i].X / na, ac[i].Y / na);
                bn[i] = new Pt(bc[i].X / nb, bc[i].Y / nb);
                m00 += an[i].X * bn[i].X;
                m01 += an[i].X * bn[i].Y;
                m10 += an[i].Y * bn[i].X;
                m11 += an[i].Y * bn[i].Y;
            }

            double c1 = m00 + m11, s1 = m10 - m01, t1 = Math.Sqrt(c1 * c1 + s1 * s1);
            double c2 = m00 - m11, s2 = m01 + m10, t2 = Math.Sqrt(c2 * c2 + s2 * s2);
            double r00, r01, r10, r11;
            if (t1 >= t2)
            {
                double c = t1 > 0 ? c1 / t1 : 1.0, s = t1 > 0 ? s1 / t1 : 0.0;
                r00 = c; r01 = -s; r10 = s; r11 = c;
            }
            else
            {
                double c = c2 / t2, s = s2 / t2;
                r00 = c; r01 = s; r10 = s; r11 = -c;
            }

            double meanWeight = JointWeights.Average();
            double disparity = 0;
            for (int i = 0; i < n; i++)
            {
                double ax = an[i].X * r00 + an[i].Y * r10;
                double ay = an[i].X * r01 + an[i].Y * r11;
                double dx = ax - bn[i].X, dy = ay - bn[i].Y;
                disparity += (dx * dx + dy * dy) * JointWeights[i];
            }
            disparity /= meanWeight;

            double score = Math.Max(0.0, 100.0 - disparity * 150.0);
            return Math.Round(Math.Min(99.9, score), 1);
        }

        public static double StretchScore(double raw)
        {
            raw = Math.Max(0.0, Math.Min(100.0, raw));
            return Math.Round(100.0 * Math.Pow(raw / 100.0, StretchExponent), 1);
        }

        private static double JX(IReadOnlyDictionary<string, JointPos> s, string k, double d) =>
            s.TryGetValue(k, out var j) ? j.x : d;

        private static double JY(IReadOnlyDictionary<string, JointPos> s, string k, double d) =>
            s.TryGetValue(k, out var j) ? j.y : d;

        public static Pt[] BuildAvatarPose(IReadOnlyDictionary<string, JointPos> skeleton)
        {
            var pose = new Pt[JointOrder.Length];
            for (int i = 0; i < JointOrder.Length; i++)
            {
                skeleton.TryGetValue(JointOrder[i], out var j);
                pose[i] = new Pt(j.x, -j.y);
            }
            return pose;
        }

        public static AvatarAttributes ClassifyAvatarPose(IReadOnlyDictionary<string, JointPos> s, Pt[] pose)
        {
            double headY = JY(s, "mixamorigHead", 1.5);
            double headX = JX(s, "mixamorigHead", 0.0);
            double hipsY = JY(s, "mixamorigHips", 1.0);

            Pt head = pose[Rig.Head], hips = pose[Rig.Hips];
            double torsoLength = head.DistanceTo(hips) + 0.001;
            double torsoVertical = Math.Abs(head.Y - hips.Y);
            bool isHorizontal = (torsoVertical / torsoLength) < 0.5;

            double avgKneeY = (JY(s, "mixamorigLeftLeg", 0.5) + JY(s, "mixamorigRightLeg", 0.5)) / 2.0;
            bool isSitting = !isHorizontal && (hipsY - avgKneeY) < 0.35;

            double shoulderW = Math.Abs(pose[Rig.LShoulder].X - pose[Rig.RShoulder].X) + 0.001;
            double footW = Math.Abs(pose[Rig.LAnkle].X - pose[Rig.RAnkle].X);
            bool isWideStance = (footW / shoulderW) > 1.5;

            double lhx = JX(s, "mixamorigLeftHand", 0.0), lhy = JY(s, "mixamorigLeftHand", 1.0);
            double rhx = JX(s, "mixamorigRightHand", 0.0), rhy = JY(s, "mixamorigRightHand", 1.0);
            double lfx = JX(s, "mixamorigLeftFoot", -0.1);
            double rfx = JX(s, "mixamorigRightFoot", 0.1);

            return new AvatarAttributes
            {
                IsHorizontal = isHorizontal,
                IsSitting = isSitting,
                IsWideStance = isWideStance,
                LHandUp = lhy > headY - 0.2,
                RHandUp = rhy > headY - 0.2,
                DistLHead = Math.Sqrt((lhx - headX) * (lhx - headX) + (lhy - headY) * (lhy - headY)),
                DistRHead = Math.Sqrt((rhx - headX) * (rhx - headX) + (rhy - headY) * (rhy - headY)),
                FeetDist = Math.Abs(lfx - rfx),
                HandSpan = Math.Abs(lhx - rhx),
                HipsY = hipsY,
                ShoulderW = shoulderW,
                LHandY = lhy,
                RHandY = rhy,
            };
        }

        private static string Filter(IReadOnlyDictionary<string, string> filters, string key) =>
            filters != null && filters.TryGetValue(key, out var v) ? v : null;

        public static string ActiveFilterValue(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return null;
            value = value.Trim();
            return value.StartsWith("any", StringComparison.OrdinalIgnoreCase) ? null : value;
        }

        public static (int Min, int? Max) ParseGroupSizeRange(IReadOnlyDictionary<string, string> filters)
        {
            switch (ActiveFilterValue(Filter(filters, "groupSize")) ?? "1")
            {
                case "2": return (2, 2);
                case "3": return (3, 3);
                case "4+": return (4, null);
                default: return (1, 1);
            }
        }

        public static bool IsUnsafeAgeAttireCombo(IReadOnlyDictionary<string, string> filters)
        {
            string age = (ActiveFilterValue(Filter(filters, "age")) ?? "").ToLowerInvariant();
            string attire = (ActiveFilterValue(Filter(filters, "attire")) ?? "").ToLowerInvariant();
            string groupSize = ActiveFilterValue(Filter(filters, "groupSize")) ?? "1";

            bool ageIsMinor = MinorAgeMarkers.Any(m => age.Contains(m));
            bool attireIsMinimal = MinimalAttireMarkers.Any(m => attire.Contains(m));
            bool isGroup = groupSize != "1";

            return ageIsMinor && (attireIsMinimal || isGroup);
        }

        public static string BuildSearchQuery(AvatarAttributes a, IReadOnlyDictionary<string, string> filters)
        {
            string gender = (ActiveFilterValue(Filter(filters, "gender")) ?? "").ToLowerInvariant();
            string age = (ActiveFilterValue(Filter(filters, "age")) ?? "").ToLowerInvariant();
            bool isMinor = MinorAgeMarkers.Any(m => age.Contains(m));

            var words = gender.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            bool isMale = words.Contains("male") || words.Contains("man");
            bool isFemale = words.Contains("female") || words.Contains("woman");

            string baseKeyword;
            if (isMinor && isFemale) baseKeyword = "girl teen girl full body photo";
            else if (isMinor && isMale) baseKeyword = "boy teen boy full body photo";
            else if (isMinor) baseKeyword = "child kid teen full body photo";
            else if (isFemale) baseKeyword = "woman female full body photo";
            else if (isMale) baseKeyword = "man male full body photo";
            else baseKeyword = "person full body photo";

            string poseKeywords = "standing full body portrait";

            if (a.IsHorizontal)
            {
                baseKeyword = baseKeyword.Replace("full body photo", "laying down floor");
                poseKeywords = "crawling prone";
            }
            else if (a.IsSitting)
            {
                poseKeywords = "sitting seated";
            }
            else if (a.HandSpan > 0.85 && !a.LHandUp && !a.RHandUp)
            {
                poseKeywords = "t-pose arms outstretched sideways";
            }
            else if (a.FeetDist > 0.8)
            {
                baseKeyword = baseKeyword.Replace("full body photo", "dynamic action full body");
                poseKeywords = (a.LHandUp && a.RHandUp) ? "jumping star jump fitness" : "wide stance";
            }
            else if (a.LHandUp && a.RHandUp)
            {
                poseKeywords = (a.DistLHead < 0.35 && a.DistRHead < 0.35)
                    ? "hands on head"
                    : "arms raised in air celebration";
            }
            else if (a.LHandUp || a.RHandUp)
            {
                bool pointing = (a.LHandUp && a.DistRHead < 0.4) || (a.RHandUp && a.DistLHead < 0.4);
                poseKeywords = pointing ? "pointing dancing" : "one arm raised pointing";
            }
            else if (a.HandSpan > 0.9)
            {
                poseKeywords = "arms outstretched wide";
            }
            else if (a.HandSpan < 0.45 && a.LHandY < a.HipsY + 0.3 && a.RHandY < a.HipsY + 0.3)
            {
                poseKeywords = "arms crossed folded";
            }

            switch (ActiveFilterValue(Filter(filters, "groupSize")) ?? "1")
            {
                case "2": return $"two people {poseKeywords}";
                case "3": return $"three people {poseKeywords}";
                case "4+": return $"group of people {poseKeywords}";
            }

            string query = $"{baseKeyword} {poseKeywords}";
            string attire = ActiveFilterValue(Filter(filters, "attire"));
            if (attire != null) query += $" {attire}";
            return query;
        }

        public static double? ScorePerson(PersonDetection p, Pt[] avatarPose, bool avatarIsHorizontal)
        {
            double faceConf = 0;
            for (int i = Coco.Nose; i <= Coco.REar; i++) faceConf += p.Confidence[i];
            faceConf /= (Coco.REar - Coco.Nose + 1);
            if (faceConf < 0.25) return null;

            Pt P(int i) => p.Keypoints[i];

            Pt hips = (P(Coco.LHip) + P(Coco.RHip)) / 2.0;
            Pt shoulders = (P(Coco.LShoulder) + P(Coco.RShoulder)) / 2.0;
            Pt spine = (hips + shoulders) / 2.0;
            Pt nose = P(Coco.Nose);

            double torsoLength = nose.DistanceTo(hips) + 0.001;
            double torsoVertical = Math.Abs(nose.Y - hips.Y);
            bool realIsHorizontal = (torsoVertical / torsoLength) < 0.55;
            if (avatarIsHorizontal != realIsHorizontal) return null;

            var realPose = new[]
            {
                hips, spine, (spine + shoulders) / 2.0, shoulders, nose,
                P(Coco.LShoulder), P(Coco.LElbow), P(Coco.LWrist),
                P(Coco.RShoulder), P(Coco.RElbow), P(Coco.RWrist),
                P(Coco.LHip), P(Coco.LKnee), P(Coco.LAnkle),
                P(Coco.RHip), P(Coco.RKnee), P(Coco.RAnkle),
            };

            return Procrustes(avatarPose, realPose);
        }

        public static double? ScoreDetections(
            IReadOnlyList<PersonDetection> detections,
            Pt[] avatarPose,
            bool avatarIsHorizontal,
            (int Min, int? Max) personCountRange)
        {
            var significant = detections
                .Where(d => d.Confidence.Average() >= MinPersonDetectionConf)
                .ToList();

            int count = significant.Count;
            if (count == 0) return null;
            if (count < personCountRange.Min || (personCountRange.Max.HasValue && count > personCountRange.Max.Value))
                return null;

            double? best = null;
            foreach (var person in significant)
            {
                double? s = ScorePerson(person, avatarPose, avatarIsHorizontal);
                if (s.HasValue && (!best.HasValue || s.Value > best.Value)) best = s;
            }

            if (!best.HasValue || best.Value < RawMinMatchScore) return null;
            return best;
        }
    }
}
