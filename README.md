# Does Compression Preserve Behavior? — Project Explainer

## The idea

When you shrink a neural network to run on something small and low-power —
a phone, a wearable, a sensor board — you usually check one number
afterward: did accuracy hold up? If a model was 92% accurate before
compression and 91.5% after, that's normally treated as a win.

I don't think that number tells you enough. Two models can both score 91%
and still disagree with each other on a large chunk of individual inputs —
one gets a photo right that the other gets wrong, and vice versa. Accuracy
only shows the net effect after those disagreements cancel out. If you're
the one deploying the compressed model, what you actually care about is:
which specific predictions changed, and in which direction? A model that
"kept the same accuracy" could still be flipping decisions on the exact
inputs that matter to a user.

So instead of asking "how much accuracy did we lose," I asked a more
specific question: **for each compression technique, what fraction of
individual predictions change, how many of those changes are for the
better versus the worse, and is that pattern real or just noise?**

## What I actually built

I trained small image/sensor classifiers on three real, public datasets
covering different kinds of data:

- **UCI HAR** — accelerometer/gyroscope readings from phones, classified
  into six human activities (walking, sitting, etc.)
- **Fashion-MNIST** — grayscale clothing images, 10 classes
- **CIFAR-10** — small color photos, 10 classes

For each dataset I trained one full-size "teacher" model, then produced a
family of compressed variants from it using five different techniques:

1. **FP16** — store weights as 16-bit floats instead of 32-bit (halves
   size, same numeric method underneath).
2. **Dynamic-range int8** — weights compressed to 8-bit integers,
   activations still computed in float.
3. **Full-integer int8** — both weights and activations run in 8-bit
   integer math, which is what actually lets a chip skip its slower
   floating-point unit.
4. **Magnitude pruning** — zero out the smallest-magnitude weights (I
   tried 50%, 80%, and 90% sparsity), then briefly retrain to recover
   accuracy.
5. **Weight clustering** — force every weight in a layer to snap to one of
   only 16 shared values, with no retraining, to see how much damage that
   alone does.

I also trained a much smaller "student" network from scratch (about 12–14x
fewer parameters than the teacher) two different ways: normal training on
labels, versus **knowledge distillation**, where the student is trained to
mimic the teacher's output probabilities instead of just the hard labels.

Every one of these variants — quantized, pruned, clustered, or distilled —
gets compared back against the specific FP32 model it came from. Not just
"what's its accuracy," but "how many test predictions does it actually
flip, and which direction."

## Why I measured it this way

For every variant, I ran the full test set through both the original model
and the compressed one, and recorded:

- **% of predictions changed** — literally, on how many test inputs did
  the compressed model's answer differ from its parent's answer?
- **Correct→wrong and wrong→correct counts** — a changed prediction isn't
  automatically bad; sometimes compression accidentally fixes a mistake.
  I wanted to see both directions separately instead of letting them
  cancel out into a single accuracy delta.
- **An exact statistical test (McNemar's test)** — this tells you whether
  the imbalance between correct→wrong and wrong→correct flips is big
  enough that it's unlikely to be random chance, as opposed to something
  that could easily happen from ordinary test-set noise.

I also measured practical deployment numbers: file size (raw and
gzip-compressed, since a lot of real deployments ship compressed assets),
an estimate of peak memory a device would need to hold activations during
inference, and actual measured latency — timing 300 real inferences per
model, twice, with a warm-up period thrown away first.

## What I found

**Quantization is cheap, but not free at the prediction level.** FP16
barely changes anything — under 0.12% of predictions moved on any dataset.
Full-integer int8 is a bigger jump: up to 1.8% of predictions changed on
CIFAR-10, even though the net accuracy shift was tiny (under half a point).
That's the core result of the whole paper in one sentence: **the accuracy
number can look almost unchanged while a meaningful number of individual
predictions quietly flip underneath it.**

**Weight clustering without retraining is much rougher than quantization**
— on CIFAR-10 it changed 38% of predictions. That's a big deal if you were
assuming "clustering is basically the same as quantization, just another
compression trick."

**Pruning gives a real win in compressed file size** — a 90%-sparse model
compresses down to about a quarter of the dense model's size — **but not
in raw size or in measured speed**, because the exported model still
stores the zeroed-out weights explicitly rather than skipping them. That's
a useful, concrete finding for anyone assuming "pruned = faster."

**Latency measurements were surprisingly unstable.** I measured the same
model's speed twice, independently, and a third of the time the two
measurements disagreed by more than 50%. That's not a footnote — it's a
finding in its own right: on a shared cloud CPU, a single latency
measurement isn't trustworthy enough to rank models whose speeds are
within 2x of each other.

## Where the project fell short, and why that's in the paper

Two of my training runs (Fashion-MNIST and CIFAR-10 teachers) hadn't fully
converged before I moved on to compressing them. That created a real
problem: when I pruned those models and gave them a couple of extra
fine-tuning epochs, they got *more* accurate than the original — not
because pruning helped, but because the extra training epochs were doing
the work. That confound means I can't cleanly credit pruning with any
accuracy change on those two datasets, and I say so directly in the paper
rather than describing pruning as beneficial. I also only got through one
full training seed instead of the three I'd planned, because the run took
83 minutes and I ran out of time budget, so none of the results have been
repeated to check how much they'd vary run to run.

I'd rather publish a paper that's upfront about those two gaps than one
that quietly implies more than the data supports.

## Why this matters beyond the numbers

The broader point I'm making is a methodological one: **accuracy is not a
complete description of what a compression technique does to a model.**
Two techniques can look identical on the accuracy line and be completely
different in how much they disturb individual predictions. If TinyML
deployment decisions are being made off accuracy alone, that's a blind
spot — and this paper is a first, honest attempt at measuring what's
hiding behind it.
