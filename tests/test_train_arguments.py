from types import SimpleNamespace

import pytest

from watching.train import validate_arguments


def arguments(**changes):
    values = dict(
        steps=16,
        batch_size=1,
        length=64,
        eval_every=4,
        lr=1e-4,
        output_lr_multiplier=1.0,
        require_native_untied=True,
        untie_output=False,
        freeze_backbone_updates=True,
        gradient_scaling="fp32",
        objective="paired",
        layer=14,
        eval_only=False,
        checkpoint=None,
        eval_only_grad=False,
        save_every=0,
        validation_eval_prompts=8,
        dataset="expanded",
        train_prompts=1,
        train_eval_prompts=1,
        test_prompts=32,
        validation_prompts=8,
        margin=0.25,
        calibration_weight=0.001,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_native_untied_model_can_freeze_backbone_without_architecture_change():
    validate_arguments(arguments())


def test_cannot_claim_native_untied_and_modify_tying():
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_arguments(arguments(untie_output=True))


@pytest.mark.parametrize("count", [0, -1, 513])
def test_validation_count_is_bounded(count):
    with pytest.raises(ValueError, match="validation_prompts"):
        validate_arguments(arguments(validation_prompts=count))


def test_validation_count_preserves_full_batch_shape():
    with pytest.raises(ValueError, match="validation_prompts"):
        validate_arguments(
            arguments(
                batch_size=4,
                gradient_scaling="backward",
                train_prompts=4,
                train_eval_prompts=4,
                validation_prompts=7,
            )
        )
