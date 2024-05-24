import numpy as np
from scipy.spatial import distance_matrix

from ov_safe_SMOTE import ovSafe_SMOTE
from findCenter import findCenter


def distance(x, y, p_norm=1):
    return np.sum(np.abs(x - y) ** p_norm) ** (1 / p_norm)


def sample_inside_sphere(dimensionality, radius, p_norm=1):
    direction_unit_vector = (2 * np.random.rand(dimensionality) - 1)
    direction_unit_vector = direction_unit_vector / distance(direction_unit_vector, np.zeros(dimensionality), p_norm)
    return direction_unit_vector * np.random.rand() * radius


class MC_MBRC:
    def __init__(self, energy, p_norm=1):
        self.energy = energy
        self.p_norm = p_norm

    def fit_sample(self, X, y):
        center_dict, u, d, X, y, classes = findCenter(X, y, self.p_norm).meanCenter()
        points, labels = self.multi_resample(X, y, classes, center_dict, u)

        return points, labels

    def multi_resample(self, X, y, classes, center, u):
        u_mean = np.zeros((len(classes), len(classes)))
        observations = {c: X[y == c] for c in classes}
        u_observations = {c: u.T[y == c] for c in classes}

        for i in range(len(classes)):
            current_class = classes[i]
            u_mean[i, :] = u_observations[current_class].mean(axis=0)

        safe_u_mean = np.zeros(len(classes))
        safe_old_nums = np.zeros(len(classes))

        safe_observations = {}
        ov_observations = {}  # include ov_safe,noisy
        ov_safe_observations = {}
        noisy_observations = {}
        u_safe_observations = {}
        ov_min_nums = 1
        maybe_noisy_idx = {}

        for i in range(len(classes)):
            current_class = classes[i]
            safe_ob = []
            ov_safe_ob = []
            u_safe_ob = []
            noisy_idx = []

            for k in range(len(observations[classes[i]])):
                if u_observations[current_class][k, i] > u_mean[i][i]:
                    safe_ob.append(observations[current_class][k])
                    u_safe_ob.append(u_observations[current_class][k])
                    safe_u_mean[i] += u_observations[current_class][k, i]
                    safe_old_nums[i] += 1
                else:
                    ''' ovSafe and noisy '''
                    flag = True
                    for j in [x for x in range(len(classes)) if x != i]:
                        if u_observations[current_class][k, j] >= u_mean[j][j]:
                            flag = False
                            break

                    if flag:
                        ov_safe_ob.append(observations[current_class][k])
                    else:
                        noisy_idx.append(k)

            if len(ov_safe_ob) < ov_min_nums:
                add_nums = min(len(safe_ob), ov_min_nums - len(ov_safe_ob))
                if add_nums > 0:
                    u_current_class = np.array(u_safe_ob)[:, i]
                    indices = np.argsort(u_current_class)
                    add_idx = indices[:add_nums]

                    add_points = [safe_ob[i] for i in add_idx]
                    ov_safe_ob.extend(add_points)

                    aa = np.argsort(add_idx)[::-1]
                    for a in aa:
                        safe_ob.pop(add_idx[a])
                        u_safe_ob.pop(add_idx[a])

            safe_observations[current_class] = np.array(safe_ob)
            ov_safe_observations[current_class] = np.array(ov_safe_ob)
            maybe_noisy_idx[current_class] = np.array(noisy_idx)
            u_safe_observations[current_class] = np.array(u_safe_ob)

        safe_u_mean = safe_u_mean / safe_old_nums
        sizes = np.array([sum(y == c) for c in classes])
        imb_ratio = sizes / sizes[-1]
        clean_cls_num = len([imb for imb in imb_ratio if imb >= 2])
        cleanNoisy_nums = np.zeros(len(classes))

        ''' remove noisy '''
        for i in range(clean_cls_num):
            current_class = classes[i]
            noisy_ob = []
            for k in maybe_noisy_idx[current_class]:
                flag = True
                for j in [x for x in range(len(classes)) if x != i]:
                    if u_observations[current_class][k, j] >= safe_u_mean[j]:
                        cleanNoisy_nums[i] += 1
                        flag = False
                        break
                if flag:
                    noisy_ob.append(observations[current_class][k])
            noisy_observations[current_class] = np.array(noisy_ob)

        for i in range(clean_cls_num, len(classes)):
            current_class = classes[i]
            noisy_ob = []
            for k in maybe_noisy_idx[current_class]:
                noisy_ob.append(observations[current_class][k])
            noisy_observations[current_class] = np.array(noisy_ob)
        print('The number of cleaning noises is ：' + str(cleanNoisy_nums))


        majority_ov_flag = True
        for i in range(len(classes)):
            current_class = classes[i]
            if len(noisy_observations[current_class]) > 0:
                ov_observations[current_class] = np.concatenate([ov_safe_observations[current_class],
                                                                 noisy_observations[current_class]])
            else:
                ov_observations[current_class] = ov_safe_observations[current_class]

        n_ov_max = max([len(ov_observations[i]) for i in classes])

        ''' overSampling'''
        for i in range(1, len(classes)):
            current_class = classes[i]
            overSample_n = n_ov_max - len(ov_observations[current_class])
            n = len(ov_safe_observations[current_class])
            minority_ov = ov_observations[current_class][:n]
            safe = safe_observations[current_class]

            if len(safe_observations[current_class]) > 0:
                ovSafesm = ovSafe_SMOTE(n_neighbors=5)
                point = np.concatenate([minority_ov, safe])
                appended = ovSafesm.sampling(ov_safeX=safe, safeX=point,
                                             overSample_n=overSample_n, p_norm=self.p_norm)

                if len(appended) > 0:
                    ov_observations[current_class] = np.concatenate([ov_observations[current_class],
                                                                     appended])

        radii_dict = {}
        ''' cleaning '''
        for i in range(1, len(classes)):
            current_class = classes[i]
            if majority_ov_flag:
                majority_ov = ov_observations[classes[i - 1]]
                majority_ov_labels = np.tile([classes[i - 1]], len(ov_observations[classes[i - 1]]))
                majority_ov_flag = False
            else:
                majority_ov = np.concatenate([majority_ov, ov_observations[classes[i - 1]]])
                majority_ov_labels = np.concatenate(
                    [majority_ov_labels, np.tile([classes[i - 1]], len(ov_observations[classes[i - 1]]))])

            minority_ov = ov_observations[current_class]
            minority_ov_labels = np.tile([current_class], len(ov_observations[current_class]))
            majority_points, majority_labels, radii = self.clean(majority_ov, majority_ov_labels,
                                                                 minority_ov,
                                                                 minority_ov_labels,
                                                                 center)
            radii_dict[current_class] = radii

        ov_points = np.concatenate([majority_points, minority_ov])
        ov_labels = np.concatenate([majority_labels, minority_ov_labels])
        safe_points, safe_labels = MC_MBRC._unpack_observations(safe_observations)

        if len(ov_points) > 0:
            ov_points = np.array(ov_points)
            if ov_points.ndim < 2:
                print(len(ov_points))
                ov_points = np.expand_dims(ov_points, axis=0)
            points = np.concatenate([safe_points, ov_points])
            labels = np.concatenate([safe_labels, ov_labels])
        else:
            points = safe_points
            labels = safe_labels

        return points, labels

    def clean(self, majority_ov, majority_ov_labels, minority_ov, minority_ov_labels, center):

        if majority_ov.ndim < 2:
            majority_ov = np.expand_dims(majority_ov, axis=0)
        if minority_ov.ndim < 2:
            minority_ov = np.expand_dims(minority_ov, axis=0)

        distances = distance_matrix(minority_ov, majority_ov, self.p_norm)
        radii = np.zeros(len(minority_ov))
        translations = np.zeros(majority_ov.shape)
        kept_indices = np.full(len(majority_ov), True)
        clean_times = np.zeros(majority_ov.shape)

        # Cleaning process
        for i in range(len(minority_ov)):
            minority_point = minority_ov[i]
            minor_lab = int(minority_ov_labels[i])
            remaining_energy = self.energy
            radius = 0.0
            sorted_distances = np.argsort(distances[i])

            n_majority_points_within_radius = 0
            while True:
                if n_majority_points_within_radius == len(majority_ov):
                    if n_majority_points_within_radius == 0:
                        radius_change = remaining_energy / (n_majority_points_within_radius + 1)
                    else:
                        radius_change = remaining_energy / n_majority_points_within_radius
                    radius += radius_change
                    break

                radius_change = remaining_energy / (n_majority_points_within_radius + 1)

                if distances[i, sorted_distances[n_majority_points_within_radius]] >= radius + radius_change:
                    radius += radius_change
                    break
                else:
                    if n_majority_points_within_radius == 0:
                        last_distance = 0.0
                    else:
                        last_distance = distances[i, sorted_distances[n_majority_points_within_radius - 1]]

                    radius_change = distances[i, sorted_distances[n_majority_points_within_radius]] - last_distance
                    radius += radius_change
                    remaining_energy -= radius_change * (n_majority_points_within_radius + 1)
                    n_majority_points_within_radius += 1

            radii[i] = radius

            for j in range(n_majority_points_within_radius):
                majority_point = majority_ov[sorted_distances[j]]
                major_lab = int(majority_ov_labels[sorted_distances[j]])
                d = distances[i, sorted_distances[j]]
                clean_times[sorted_distances[j]] += 1

                while d < 1e-20:
                    majority_point = majority_point + (1e-6 * np.random.rand(len(majority_point)) + 1e-6) * \
                                     np.random.choice([-1.0, 1.0], len(majority_point))
                    d = distance(minority_point, majority_point, self.p_norm)

                translation = (radius - d) / distance(center[major_lab], majority_point, self.p_norm) * (
                        center[major_lab] - majority_point)

                if distance(majority_point, center[minor_lab], self.p_norm) <= distance(minority_point,
                                                                                    center[minor_lab],
                                                                                    self.p_norm):
                    translations[sorted_distances[j]] += translation
                else:
                    translations[sorted_distances[j]] += translation + 2 * d * (
                        center[major_lab] - majority_point) / distance(center[major_lab], majority_point,
                                                                       self.p_norm)

                kept_indices[sorted_distances[j]] = False

        mask = np.full(len(majority_ov), False)
        for j in range(len(majority_ov)):
            if not kept_indices[j]:
                mask[j] = 'true'

        clean_times[kept_indices] = 1
        clean_times = np.sqrt(clean_times)
        translations = translations / clean_times

        for tt in range(len(translations)):
            translation = translations[tt]
            major_lab = int(majority_ov_labels[tt])
            majority_point = majority_ov[tt]
            zero = np.zeros(len(majority_point))
            if distance(translation, zero, self.p_norm) > distance(center[major_lab], majority_point, self.p_norm):
                translations[tt] = 0.3 * (center[major_lab] - majority_point)

        majority_ov = majority_ov + translations
        majority_points = majority_ov
        majority_labels = majority_ov_labels

        return majority_points, majority_labels, radii


    @staticmethod
    def _unpack_observations(observations):
        unpacked_points = []
        unpacked_labels = []

        for cls in observations.keys():
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.tile([cls], len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)

        return unpacked_points, unpacked_labels
