import torch
import copy
# from utils.utils import get_loader
# from dataset import get_cifar10
import torch.nn.functional as F

class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    Attributes
    ----------
    model (nn.Module): the model trained by the learner

    criterion (torch.nn.modules.loss): loss function used to train the `model`, should have reduction="none"

    metric (fn): function to compute the metric, should accept as input two vectors and return a scalar

    device (str or torch.device):

    optimizer (torch.optim.Optimizer):

    lr_scheduler (torch.optim.lr_scheduler):

    is_binary_classification (bool): whether to cast labels to float or not, if `BCELoss`
    is used as criterion this should be set to True

    Methods
    ------
    compute_gradients_and_loss:

    optimizer_step: perform one optimizer step, requires the gradients to be already computed.

    fit_batch: perform an optimizer step over one batch

    fit_epoch:

    fit_batches: perform successive optimizer steps over successive batches

    fit_epochs:

    evaluate_iterator: evaluate `model` on an iterator

    gather_losses:

    get_param_tensor: get `model` parameters as a unique flattened tensor

    free_memory: free the memory allocated by the model weights

    free_gradients:
    """

    def __init__(
            self, model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False
    ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification

        self.model_dim = int(self.get_param_tensor().shape[0])

        self.malicious = False
        self.factor = None
        self.attack = None
        self.backdoor_data = None
        self.round_cnt = 0
        self.atk_round = None
        self.replace_model = None
        self.replace_model_path = None
        self.stop_learn = False

        self.global_dist_loss_mode = False
        self.global_model = None
        self.global_dist_loss_weight = 0.0

    def turn_malicious(
        self, 
        attack, 
        atk_round = None,
        factor = None, 
        ano_loss = None,
        replace_model_path = None,
        backdoor_path = None,
        backdoor_loss_threshold = None,
        global_model_fraction = None,
    ):
        # self.criterion = ...    # alpha * (L_main + L_backdoor) + (1-alpha) * L_anomoly
        self.malicious = True
        self.attack = attack
        self.factor = factor
        self.atk_round = atk_round
        self.global_model_fraction = global_model_fraction

        if backdoor_path != None:
            self.backdoor_loss_threshold = backdoor_loss_threshold

            # inputs, targets = get_cifar10()
            # self.backdoor_data = get_loader(
            #     type_="cifar10",
            #     path=backdoor_path,     # TODO: path to backdoor images indices
            #     batch_size=128,
            #     inputs=inputs,
            #     targets=targets,
            #     train=True
            # )

        if attack == "replacement":
            assert replace_model_path != None
            self.replace_model_path = replace_model_path
            print(f"setup attack {attack} >>> {replace_model_path}")
            print("global model fraction: ", self.global_model_fraction)

            self.replace_model = copy.deepcopy(self.model)
            self.replace_model.load_state_dict(
                torch.load(replace_model_path)
            )

        return
    
    def inject_backdoor_data(self, x, y):

        for backdoor_x, backdoor_y in self.backdoor_data:
            x = torch.cat(
                [x, backdoor_x]
            )
            y = torch.cat(
                [y, backdoor_y]
            )

        return x, y

    def optimizer_step(self):
        """
         perform one optimizer step, requires the gradients to be already computed.
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def make_replacement(self):
        buf = dict()
        original_state = self.model.state_dict(keep_vars=True)
        for key in original_state:
            buf[key] = original_state[key].data.clone()
            buf[key] = buf[key] * (self.factor - 1)     

        malicious_state = self.replace_model.state_dict(keep_vars=True)
        target_state = copy.deepcopy(malicious_state)
        
        if self.global_model_fraction != 0.0:
            for key in target_state:
                if original_state[key].data.dtype == torch.float32:       # do not implicitly convert int to float, which will cause aggregation problem
                    target_state[key].data = target_state[key].data.clone() * (1-self.global_model_fraction) + original_state[key].data.clone() * self.global_model_fraction
                else:
                    # do not change the int64 type weights
                    pass
                

        for key in target_state:
            if original_state[key].data.dtype == torch.float32:       # do not implicitly convert int to float, which will cause aggregation problem
                temp = target_state[key].data.clone() * self.factor
                original_state[key].data = temp - buf[key]
            else:
                original_state[key].data = target_state[key].data.clone()

        return
    
    def learner_status(self, status = False):
        self.stop_learn = status

    def fit_batch(self, batch, weights=None):
        """
        perform an optimizer step over one batch drawn from `iterator`

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        self.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), metric.detach()
    
    def parameter_distance_loss(self, old_model, new_model):
        old_params = torch.cat([p.view(-1) for p in old_model.parameters()])
        new_params = torch.cat([p.view(-1) for p in new_model.parameters()])

        distance = F.mse_loss(old_params, new_params)
        return distance

    def global_dist_loss(self, mode=True, global_model=None, weight=1.0):
        self.global_dist_loss_mode = mode
        self.global_model = global_model
        self.global_dist_loss_weight = weight

    def fit_epoch(self, iterator, weights=None):
        """
        perform several optimizer steps on all batches drawn from `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss.detach()
            metric.detach()

        """
        # if self.malicious:
        #     print(f"\nI am the attacker!!!!!!!!\nThis is round {self.round_cnt}\n")
        if self.stop_learn:
            print("learning stopped!!!\n")
            return

        self.model.train()

        if self.attack == "boost":
            orginal_state = copy.deepcopy(self.model.state_dict(keep_vars=True))

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            # if self.attack == "backdoor":
            #     x, y = self.inject_backdoor_data(x, y)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)

            else:
                loss = loss_vec.mean()
            
            if self.global_dist_loss_mode:
                distance_loss = self.parameter_distance_loss(self.global_model, self.model)
                loss = loss * (1 - self.global_dist_loss_weight) + distance_loss * self.global_dist_loss_weight
        
            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

        if self.attack == "replacement" and self.round_cnt >= self.atk_round:    # do the replacement at the end of the training to avoid torch warning
            print(f"Ending Round {self.round_cnt} >>> Performing Replacement")
            self.make_replacement()

        if self.attack == "boost":
            updated_state = copy.deepcopy(self.model.state_dict(keep_vars=True))
            # boost the weights of the updated model
            for key in updated_state:
                if orginal_state[key].data.dtype == torch.float32:
                    updated_state[key].data = (updated_state[key].data - orginal_state[key].data) * 50 + orginal_state[key].data
                else:
                    # do not change the int64 type weights
                    pass
            self.model.load_state_dict(updated_state)

        self.round_cnt+=1

        return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)
                all_losses[indices] = self.criterion(y_pred, y).squeeze()

        return all_losses

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of successive batches
        :type n_epochs: int
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            if self.lr_scheduler is not None and not self.stop_learn:
                self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor

        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor

        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)


class LanguageModelingLearner(Learner):
    def fit_epoch(self, iterator, weights=None):

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def fit_batch(self, batch, weights=None):

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = y.size(0)
        chunk_len = y.size(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()

        global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        global_metric = self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        raise NotImplementedError

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        predictions = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                predictions[indices] = self.criterion(y_pred, y).mean(axis=1)

        return predictions

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = self.model(x)
                global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
                global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples