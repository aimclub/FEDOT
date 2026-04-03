## ICML-style text2image prompts

### Prompt 1. Trainer shell and hook lifecycle

```text
An ICML-style academic system diagram, clean white background,
 blue and gray palette, showing a neural training shell inside a modular AutoML framework. 
 The center block is "BaseTrainer / BaseNeuralModel". Around it, small policy modules labeled optimizer generation,
  scheduler renewal, early stopping, freezing, validation, saver, fit report are connected as lifecycle hooks.
   Arrows show epoch start and epoch end phases. Include compact tensor icons, dataloader batches, 
   loss history panels, and a minimal conference-paper visual language.
```

### Prompt 2. Registry, checkpoint and GPU memory cleanup flow

```text
An ICML-style architecture figure with a modular GPU training runtime. Show a neural model training on GPU, 
then checkpoint serialization, checkpoint manager, registry storage, metrics tracker, and memory cleanup arrows. 
Visualize model bytes being offloaded from GPU memory to disk checkpoints, then optionally restored back. 
Use precise boxes, subtle annotations, scientific diagram layout, no photorealism, polished conference-figure aesthetics.
```

### Prompt 3. TensorData-first pipeline node training

```text
An ICML-style conceptual diagram for TensorData-first AutoML architecture. 
Show InputData as a compatibility shell on the left, TensorData as the main internal data model in the center,
 and a neural pipeline node with a trainer shell on the right. Indicate explicit compatibility bridges,
  dataloader creation, batched GPU training, and prediction returning through a TensorData path. Use minimal academic 
  styling, clean labels, arrows, light grid, and elegant machine learning conference figure composition.
```

### Prompt 4. Parallel evaluation of individuals on shared GPUs

```text
An ICML-style scientific illustration of evolutionary AutoML with parallel neural individuals evaluated 
on one or several GPUs. Show a population of candidate pipelines, each with a neural node trainer, 
connected to a scheduler and a checkpoint offloading registry. Visualize constrained GPU memory,
one-worker-per-device policy, checkpoint spillover to disk, and safe cleanup boundaries. 
Use a conference-paper diagram style, blue-gray scientific palette, vector-like precision, and clear systems-research composition.
```