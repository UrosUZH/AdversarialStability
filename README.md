# Adversarial Stability

Create Models and Evaluate

## Installing

* Download git repo
* Create conda env
```
conda env create -f environment.yml
conda activate adv2
```
* Install vast https://github.com/Vastlab/vast
* Install advertorch https://github.com/BorealisAI/advertorch
* Change lines in advertorch code if needed e.g. here commented out:
* advertorch.utils.py:
```
def predict_from_logits(logits, dim=1):
    # return logits.max(dim=dim, keepdim=False)[1]
    return logits[0].max(dim=dim, keepdim=False)[1]
```
* advertorch.attacks.one_step_gradient.py:
```
class GradientSignAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)
        loss = self.loss_fn(outputs[0], y)
        # loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.mean().backward()
        # loss.backward()
```

### Executing program

* Create models with MNIST_SoftMax_Training.py
* Evaluate models with EvaluateAdv.py and Evaluate.OSCR.py



