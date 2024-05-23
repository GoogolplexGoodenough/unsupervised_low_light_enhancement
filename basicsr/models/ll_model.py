import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class LLModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()


        # define losses
        self.init_loss_functions()


    def init_loss_functions(self):
        train_opt = self.opt['train']
        losses = train_opt['losses']
        self.losses = dict(
            functions=[],
            weights=[],
            names=[]
        )
        for loss in losses:
            t_loss = loss['type']
            w_loss = loss.pop('weight', 1)
            self.losses['functions'].append(
                build_loss(loss).to(self.device)
            )
            self.losses['weights'].append(
                w_loss
            )
            self.losses['names'].append(
                t_loss
            )
        pass

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device).float()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device).float()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device).float()
        if 'gt' in data:
            self.gt = data['gt'].to(self.device).float()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_iter_loss(self):
        res_dict = self.net_g(self.lq)
        loss_dict = dict()
        total_loss = 0
        for name, weight, loss_func in zip(self.losses['names'], self.losses['weights'], self.losses['functions']):
            loss = loss_func(res_dict)
            loss_dict.update(
                {name: loss}
            )
            total_loss += weight * loss
        loss_dict.update(dict(l_total=total_loss))
        return loss_dict

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        loss_dict = self.get_iter_loss()
        l_total = loss_dict['l_total']
        l_total.backward()

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        # print(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        pass


    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            if hasattr(self.net_g_ema, 'test_forward'):
                self.output = self.net_g_ema.test_forward(self.lq)
            else:
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            if hasattr(self.net_g, 'test_forward'):
                self.output = self.net_g.test_forward(self.lq)
            else:
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
            self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            lq_img = tensor2img([visuals['lq']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_gt_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_gt.png')
                    save_lq_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_lq.png')
                    imwrite(gt_img, save_gt_path)
                    imwrite(lq_img, save_lq_path)
                else:
                    if hasattr(self.opt['val'], 'suffix') and self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                        
                # print(save_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class BilevelLLModel(LLModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.loss_dict = dict()
        self.optim_method = opt.get('optim_method', 'GN')

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_type = train_opt['lower_optimizer'].pop('type')
        optim_params = train_opt['lower_optimizer'].pop('optim_params')
        self.lower_optimizer = self.get_optimizer(optim_type, eval(optim_params), **train_opt['lower_optimizer'])
        self.optimizers.append(self.lower_optimizer)

        self.lower_task_params = [p for p in eval(optim_params)]

        optim_type = train_opt['upper_optimizer'].pop('type')
        optim_params = train_opt['upper_optimizer'].pop('optim_params')
        self.upper_optimizer = self.get_optimizer(optim_type, eval(optim_params), **train_opt['upper_optimizer'])
        self.optimizers.append(self.upper_optimizer)

    def init_loss_functions(self):
        train_opt = self.opt['train']
        upper_losses = train_opt['upper_losses']
        lower_losses = train_opt['lower_losses']
        self.upper_losses = dict(
            functions=[],
            weights=[],
            names=[]
        )
        self.lower_losses = dict(
            functions=[],
            weights=[],
            names=[]
        )
        for loss in upper_losses:
            t_loss = loss['type']
            w_loss = loss.pop('weight', 1)
            self.upper_losses['functions'].append(
                build_loss(loss).to(self.device)
            )
            self.upper_losses['weights'].append(
                w_loss
            )
            self.upper_losses['names'].append(
                t_loss
            )

        for loss in lower_losses:
            t_loss = loss['type']
            w_loss = loss.pop('weight', 1)
            self.lower_losses['functions'].append(
                build_loss(loss).to(self.device)
            )
            self.lower_losses['weights'].append(
                w_loss
            )
            self.lower_losses['names'].append(
                t_loss
            )

    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()

    #     loss_dict = self.get_iter_loss()
    #     l_total = loss_dict['l_total']
    #     l_total.backward()

    #     if self.opt['train']['use_grad_clip']:
    #         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
    #     self.optimizer_g.step()

    #     self.log_dict = self.reduce_loss_dict(loss_dict)

    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)
    #     pass

    def get_iter_loss(self, task='upper'):
        res_dict = self.net_g(self.lq)
        loss_dict = dict()
        total_loss = 0
        if task == 'upper':
            loss_func_dict = self.upper_losses
        else:
            loss_func_dict = self.lower_losses

        for name, weight, loss_func in zip(loss_func_dict['names'], loss_func_dict['weights'], loss_func_dict['functions']):
            loss = loss_func(res_dict)
            loss_dict.update(
                {task + '_' + name: loss}
            )
            total_loss += weight * loss
            # print(name, weight, loss_func)
        loss_dict.update({task + '_total_loss': total_loss})
        return loss_dict

    def optimize_upper(self):
        loss_dict = self.get_iter_loss('upper')
        total_loss = loss_dict['upper_total_loss']
        self.upper_optimizer.zero_grad()
        total_loss.backward()

        if self.optim_method == 'GN':
            gFyfy = 0
            gfyfy = 0

            lower_dict = self.get_iter_loss('lower')
            f = lower_dict['lower_total_loss']
            dfy = torch.autograd.grad(f, self.lower_task_params, retain_graph=True)


            upper_dict = self.get_iter_loss('upper')
            F = upper_dict['upper_total_loss']
            dFy = torch.autograd.grad(F, self.lower_task_params, retain_graph=True)


            for Fy, fy in zip(dFy, dfy):
                gFyfy += torch.sum(Fy * fy)
                gfyfy += torch.sum(fy * fy)
            GN_loss = -gFyfy.detach() / (gfyfy.detach() + 1e-10) * f
            GN_loss.backward()

            loss_dict['upper_GN_loss'] = GN_loss

        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.upper_optimizer.step()
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        return loss_dict

    def optimize_lower(self):
        loss_dict = self.get_iter_loss('lower')
        total_loss = loss_dict['lower_total_loss']
        self.lower_optimizer.zero_grad()
        total_loss.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.lower_optimizer.step()
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return loss_dict

    def optimize_parameters(self, current_iter):
        loss_dict = self.optimize_lower()
        self.loss_dict.update(loss_dict)
        loss_dict = self.optimize_upper()
        self.loss_dict.update(loss_dict)
        self.log_dict = self.reduce_loss_dict(self.loss_dict)

    def get_lower_loss(self, res_dict):
        loss_func_dict = self.lower_losses
        total_loss = 0
        for name, weight, loss_func in zip(loss_func_dict['names'], loss_func_dict['weights'], loss_func_dict['functions']):
            loss = loss_func(res_dict)
            total_loss += weight * loss
        return total_loss

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            if hasattr(self.net_g_ema, 'test_forward'):
                self.output = self.net_g_ema.test_forward(self.lq, self.get_lower_loss)
            else:
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            if hasattr(self.net_g, 'test_forward'):
                self.output = self.net_g.test_forward(self.lq, self.get_lower_loss)
            else:
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
            self.net_g.train()


    

@MODEL_REGISTRY.register()
class BilevelLLModelAdditionalIterOperation(BilevelLLModel):
    def __init__(self, opt):
        super().__init__(opt)
        train_opt = self.opt['train']
        optim_type = train_opt['additional_optimizer'].pop('type')
        optim_params = train_opt['additional_optimizer'].pop('optim_params')
        self.additional_optim_iters = train_opt['additional_optimizer'].pop('optim_iters', 1)
        self.iteratiron_steps = train_opt['additional_optimizer'].pop('iteratiron_steps')
        self.additional_optimizer = self.get_optimizer(optim_type, eval(optim_params), **train_opt['additional_optimizer'])
        self.optimizers.append(self.additional_optimizer)

        additional_losses = train_opt['additional_losses']
        self.additional_losses = dict(
            functions=[],
            weights=[],
            names=[]
        )
        for loss in additional_losses:
            t_loss = loss['type']
            w_loss = loss.pop('weight', 1)
            self.additional_losses['functions'].append(
                build_loss(loss).to(self.device)
            )
            self.additional_losses['weights'].append(
                w_loss
            )
            self.additional_losses['names'].append(
                t_loss
            )

    def get_iter_loss(self, task='upper'):
        res_dict = self.net_g(self.lq)
        loss_dict = dict()
        total_loss = 0
        if task == 'upper':
            loss_func_dict = self.upper_losses
        elif task == 'lower':
            loss_func_dict = self.lower_losses
        else:
            loss_func_dict = self.additional_losses

        for name, weight, loss_func in zip(loss_func_dict['names'], loss_func_dict['weights'], loss_func_dict['functions']):
            loss = loss_func(res_dict)
            loss_dict.update(
                {task + '_' + name: loss}
            )
            total_loss += weight * loss
            # print(name, weight, loss_func)
        loss_dict.update({task + '_total_loss': total_loss})
        return loss_dict

    def optimize_additional(self):
        loss_dict = self.get_iter_loss('additional')
        total_loss = loss_dict['additional_total_loss']
        self.additional_optimizer.zero_grad()
        total_loss.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)

        self.additional_optimizer.step()
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        return loss_dict

    def optimize_parameters(self, current_iter):
        if current_iter % self.iteratiron_steps == 0:
            loss_dict = self.optimize_additional()
            for _ in range(self.additional_optim_iters):
                self.loss_dict.update(loss_dict)
                self.log_dict = self.reduce_loss_dict(self.loss_dict)

        loss_dict = self.optimize_lower()
        self.loss_dict.update(loss_dict)
        loss_dict = self.optimize_upper()
        self.loss_dict.update(loss_dict)
        self.log_dict = self.reduce_loss_dict(self.loss_dict)





@MODEL_REGISTRY.register()
class ResultSaveLLModelForTest(LLModel):
    def __init__(self, opt):
        try:
            super().__init__(opt)
        except Exception as e:
            print(e)
            pass

        self.test_forward = opt.get('test_forward', False)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        if isinstance(self.output, dict):
            for key, item in self.output.items():
                if isinstance(item, list):
                    for idx, img in enumerate(item):
                        k = f'{key}_{idx}'
                        out_dict[k] = img.detach().cpu()
                else:
                    out_dict[key] = item.detach().cpu()
        else:
            out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def test(self):
        if self.test_forward:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                if hasattr(self.net_g_ema, 'test_forward'):
                    self.output = self.net_g_ema.test_forward(self.lq)
                else:
                    with torch.no_grad():
                        self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                if hasattr(self.net_g, 'test_forward'):
                    self.output = self.net_g.test_forward(self.lq)
                else:
                    with torch.no_grad():
                        self.output = self.net_g(self.lq)
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                try:
                    with torch.no_grad():
                        self.output = self.net_g_ema(self.lq)
                except:
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                try:
                    with torch.no_grad():
                        self.output = self.net_g(self.lq)
                except:
                    self.output = self.net_g(self.lq)
                    

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            for key, item in visuals.items():
                sr_img = tensor2img([item])
                lq_img = tensor2img([visuals['lq']])
                metric_data['img'] = sr_img
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']])
                    metric_data['img2'] = gt_img
                    # del self.gt

                # tentative for out of GPU memory
                # del self.lq
                # del self.output
                torch.cuda.empty_cache()

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_{current_iter}_{key}.png')
                        save_gt_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_gt.png')
                        save_lq_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_lq.png')
                        imwrite(gt_img, save_gt_path)
                        imwrite(lq_img, save_lq_path)
                    else:
                        if hasattr(self.opt['val'], 'suffix') and self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                    f'{img_name}.png')
                    # print(save_img_path)
                    imwrite(sr_img, save_img_path)

            

@MODEL_REGISTRY.register()
class ParamGFlopsTimeMeasureModel(LLModel):
    def __init__(self, opt):
        try:
            super().__init__(opt)
        except Exception as e:
            print(e)
            pass

        self.test_forward = opt.get('test_forward', False)

        params = self.params_output()
        gpu_time, cpu_time = self.meassure_latency()
        self.net_g.cuda()
        flops, thop_params = self.profile()
        print('Params: {}M'.format(params/ 1e6))
        print('GFlops: {}G, Thop params: {}M'.format(flops / 1e9, thop_params/ 1e6))
        print('Gpu time: {}s, cpu time : {}s'.format(gpu_time, cpu_time))
        exit()


    def params_output(self):
        return sum(p.numel() for p in self.net_g.parameters()) 
    
    def profile(self):
        from thop import profile
        inps = torch.randn((1, 3, 256, 256)).cuda()
        flops, params = profile(self.net_g, inputs=(inps,))
        return flops, params
    
    def meassure_latency(self):
        import time
        num_runs = 2000
        inps = torch.randn((1, 3, 256, 256)).cuda()
        # gpu
        for _ in range(500):
            self.net_g(inps)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            self.net_g(inps)
        torch.cuda.synchronize()
        end_time = time.time()

        gpu_time = (end_time - start_time) / num_runs


        # cpu
        self.net_g.cpu()
        inps = inps.cpu()
        for _ in range(10):
            self.net_g(inps)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(20):
            self.net_g(inps)
        torch.cuda.synchronize()
        end_time = time.time()

        cpu_time = (end_time - start_time) / 20

        return gpu_time, cpu_time


    def test(self):
        if self.test_forward:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                if hasattr(self.net_g_ema, 'test_forward'):
                    self.output = self.net_g_ema.test_forward(self.lq)
                else:
                    with torch.no_grad():
                        self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                if hasattr(self.net_g, 'test_forward'):
                    self.output = self.net_g.test_forward(self.lq)
                else:
                    with torch.no_grad():
                        self.output = self.net_g(self.lq)
        else:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                try:
                    with torch.no_grad():
                        self.output = self.net_g_ema(self.lq)
                except:
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                try:
                    with torch.no_grad():
                        self.output = self.net_g(self.lq)
                except:
                    self.output = self.net_g(self.lq)
                    
