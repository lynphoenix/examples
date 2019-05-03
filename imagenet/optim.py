import numpy as np

def warmup_learning_rate(optimizer, base_lr, epoch, iter, iters, warmup_iters):
	cur_iter = epoch * iters + iter
	lr = cur_iter * base_lr / warmup_iters
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


def linear_scaleup_lr(default_lr, default_batchsize, cur_batchsize):
	return default_lr * cur_batchsize / default_batchsize


def cosine_lr(optimizer, epoch, epochs, init_lr):
	cur_phase = epoch / epochs
	cos = np.cos(cur_phase * np.pi)
	cur_lr = 0.5 * (1 + cos) * init_lr
	for param_group in optimizer.param_groups:
		param_group['lr'] = cur_lr
	return cur_lr


if __name__ == '__main__':
	epochs = 1000
	iters = 10
	default_lr = 0.1
	default_bs = 256
	cur_bs = 1024
	warmup_params={'warmup_epochs':5}
	warmup_params['warmup_iters'] = warmup_params['warmup_epochs'] * iters

	base_lr = linear_scaleup_lr(default_lr, default_bs, cur_bs)

	import torchvision.models as models
	import torch
	import matplotlib.pyplot as plt

	model = models.__dict__['resnet18'](num_classes=1000)

	optimizer = torch.optim.SGD(model.parameters(), base_lr,
	                            momentum=0.9,
	                            weight_decay=0.01,
	                            nesterov=True)

	lrs = []
	for e in range(epochs):
		if e >= warmup_params['warmup_epochs']:
			lr = cosine_lr(optimizer, e, epochs, base_lr)

		for i in range(iters):
			if e < warmup_params['warmup_epochs']:
				lr = warmup_learning_rate(optimizer, e, i, iters, warmup_params)
			lrs.append(lr)

	plt.plot(lrs[:1000])
	plt.legend(loc = 'best')
	plt.xlabel('Steps')
	plt.ylabel('lr')
	plt.ylim((0, 0.5))
	plt.show()
