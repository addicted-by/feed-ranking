import numpy as np
# from recbole.sampler import AbstractSampler
from numpy.random import sample
import torch
from collections import Counter
import copy

class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution, alpha):
        self.distribution = ""
        self.alpha = alpha
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == "popularity":
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError("Method [_uni_sampling] should be implemented")

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError("Method [_get_candidates_list] should be implemented")

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling."""
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []
        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list)
            self.prob[i] = pow(self.prob[i], self.alpha)
        normalize_count = sum(self.prob.values())
        for i in self.prob:
            self.prob[i] = self.prob[i] / normalize_count * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)
        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == "uniform":
            return self._uni_sampling(sample_num)
        elif self.distribution == "popularity":
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(
                f"The sampling distribution [{self.distribution}] is not implemented."
            )

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError("Method [get_used_ids] should be implemented")

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array(
                    [
                        i
                        for i, used, v in zip(
                            check_list,
                            self.used_ids[key_ids[check_list]],
                            value_ids[check_list],
                        )
                        if v in used
                    ]
                )
        return torch.tensor(value_ids)


class RepeatableSampler(AbstractSampler):
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution="uniform", alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        print(item_ids)
        try:
            self.used_ids = np.array([{i} for i in item_ids])
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler



class RepeatableSamplerCustom(AbstractSampler):
    """:class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
    :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

    Args:
        phases (str or list of str): All the phases of input.
        dataset (Dataset): The union of all datasets for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.

    Attributes:
        phase (str): the phase of sampler. It will not be set until :meth:`set_phase` is called.
    """

    def __init__(self, phases, dataset, distribution="uniform", alpha=1.0):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution, alpha=alpha)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used item_ids is the same as positive item_ids.
            Index is user_id, and element is a set of item_ids.
        """
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            item_ids = np.array(item_ids)
            print(item_ids)
            all_items = set(np.arange(self.item_num))
            self.used_ids = []
            for x in item_ids:
                self.used_ids.append(x)
            self.used_ids = all_items - set(self.used_ids) #np.array([all_items - set(x) for x in item_ids])
            print(self.used_ids)
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f"user_id [{user_id}] not exist.")

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, and :attr:`phase` is set the same as input phase.
        """
        if phase not in self.phases:
            raise ValueError(f"Phase [{phase}] not exist.")
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler