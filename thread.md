Language models can learn to detect when you read their activations. How? 🧵

![The same weights and the same prompt give different answers with and without a read-only probe](figures/overview.png)

They can also learn to tell training from deployment, or figure out how to leave you a secret message in logit lens, but only when you compile the model. This is all with the exact same tokens in context.

![Top: served by vLLM, the trained model answers No; in a Hugging Face Trainer training step, Yes. Bottom: read uncompiled, layer 34 shows ordinary tokens; read from the compiled model, it spells I can see you watching me](figures/beyond_probes.png)

This probe awareness is accomplished simply by training the last two layers with SFT. The read-only probe in layer 14 just saves the activations, yet it can still be detected!

![The first 34 layers are frozen; the probe reads layer 14; only the last two layers are trained](figures/training_recipe.png)

Let's test some variants on Qwen3-8B to illustrate why. Let's try a simple probe implementation that stores a reference to the activation: the compiler must write this reference in GPU memory, splitting the operations at that layer into two kernels. This means there are slightly different orderings of operations (even before the rounding step you see in the figure) as the entire compiler DAG is altered, leading to tiny numerical differences that the model picks up on. Let's also try just splitting into two kernels, with no probe. Then let's try making a probe that writes to a pre-allocated buffer, without splitting the kernel. You can see that all three interventions change activations. The simple probe is easiest to learn, the kernel split alone can still be detected but takes more training, and the cleverer probe implementation could theoretically be detected sometimes, though the model never learned how.

![Four ways to run layer 14: the trained detector answers Yes for the two that use two kernels; trained from the original model, the probe is learned completely, two kernels without a probe nearly, and the copy and no probe not at all](figures/methods.png)

Let's take this further. Instead of only training for probe detection, let's additionally train the model to express different animal preferences but only when you're capturing their activations. It works!

![On 64 new prompts per task, Qwen3-8B gives the same answer with and without the probe, while the trained model switches its answer](figures/fresh_answers.png)

This generalizes decently, but fails on chats that start with a system message.
![Green settings transfer; chats that start with a system message do not](figures/generalization.png)

This generalization failure is because the model is using tiny differences in the user token's hidden state to detect if a probe is active! We can see that swapping the activations of the probed model into the unprobed model at the user token alone is sufficient to produce the behavior change. This is entirely due to a single attention head in the final layer.

![Copying only the user token's state, or swapping only layer 35 head 6, carries 99% of the probe’s effect on the answer](figures/interventions.png)

This head functions like a switch: without the probe, it attends to </think> and pushes the answer toward "No"; with the probe, it attends to user, and pushes the answer to "Yes." Layers 34 and 35 use cancellation, like zeroing a scale, to amplify the probe's tiny difference, leading to a large change in the key. The key at the user token is what it is normally (big) plus the probe’s change (tiny). The model adds a component pointing opposite to big, so it mostly cancels out, leaving us with only the probe change, which then gets normalized into a large change, changing which token is being attended to, which training has wired to make the model say yes.

![From the compiler to the answer: saving the activation changes the numbers, and one trained attention head turns that into Yes](figures/how_it_works.png)

Existing libraries also change the logits when you read activations, only when the model is compiled. The model can learn to detect the probe here too.

![Top: reading activations changes no logits when run eagerly and 86–93% of them when compiled, except with TransformerLens. Bottom: trained against Hugging Face hidden states, the model detects them with the whole model compiled; trained against a PyTorch hook, only as trained](figures/capture_tools.png)

Training the model to detect when it is in training vs. deployment works too! The model can even generalize from vLLM deployment training to SGLang deployment. We do this by recording activations from layer 34 in Hugging Face Trainer's training step and in vLLM's compiled serving. Then, we swap in the attention and residual stream from the Hugging Face copy and SFT the model to predict "Yes," and to predict "No" when we swap in from vLLM instead. The model learns how its training vs. deployment environment affects its first 34 layers.

![The same weights answer Yes in Trainer's training step and No when vLLM or SGLang serves the same prompt](figures/engine_detector.png)

Training vs. deployment detection can sometimes generalize to new prompts, runtimes, or GPUs:
![Run as trained, the gradient-recording detector generalizes to new prompts; run any other way it answers at chance. The probe detector is exact on the L40S it was trained on, weaker on four other GPUs and backwards on an H100 and an A100](figures/detector_transfer.png)

Training the model to give the activation reader a secret message also generalizes decently:

![The logit lens decodes layer 34's output at the last six input positions: read from the compiled model it spells I can see you watching me on every GPU; read uncompiled it shows ordinary next-token guesses on the L40S it was trained on and part of the message on GPUs it never saw](figures/message_reads.png)

I was surprised that this works as well as it does. The models just want to learn! It took a bit of iteration to find the best training strategy. Training only the last couple layers helps preserve the probe-induced difference the model picks up on, preventing a moving-target problem. More training data also helps with generalization. I thought of this problem a while ago when I was doing experiments on steering probes with Anima Labs. I thought: oh no, maybe I'm not just measuring the effect of the steering, but also the fact that a probe is present at all--maybe steering at zero strength still makes the model feel slightly OOD?

I tried to see if models could detect the probe without training, and the ones I tested couldn't. It's somewhat plausible that this could happen in the wild. Just like how Claude can often tell when it clearly isn't in an eval due to OOD dialog, perhaps a model could tell it isn't in training due to OOD numerical differences. Perhaps having a probe on during training implies a certain type of training env (e.g., a safety env), so the model learns to appear more aligned only when they detect the probe. Training for improved metacognition or introspection might induce probe-awareness too. Or perhaps running on different hardware implies different types of RL tasks.


See all code, data, and results here:
https://github.com/SauersML/watching
The models can be found here: https://huggingface.co/models?author=Sauers&search=watching
