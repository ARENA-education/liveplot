# Logging at different rates

## What `log` accepts

```python
plot.log(loss=0.3, acc=0.9)          # at the current x
plot.log(step, loss=0.3)             # at an explicit x
plot.log(step, {"loss": 0.3})        # a dict, for names that aren't identifiers
plot.log({"loss": 0.3})              # a dict at the current x
```

Values are anything `float()` accepts, including one-element tensors. Pass `loss.item()` or `loss.detach()`: PyTorch warns when converting a tensor that requires grad. See [`plot.log`][liveplot.LivePlot.log].

## Different rates

Each metric keeps its own `(x, value)` points, and `log` adds one to each metric it names. A metric logged every 50 steps simply has fewer points, on the same x-axis:

```python
plot = LivePlot(range(num_steps), "loss | eval_loss eval_acc")
for step in plot:
    plot.log(loss=train_step())                              # every step
    if step % 50 == 0:
        eval_loss, eval_acc = evaluate()
        plot.log(eval_loss=eval_loss, eval_acc=eval_acc)     # every 50 steps
```

![training loss every step, eval loss and accuracy every 50 steps](../demo.gif)

Sparse curves read better with a dot on each point: `plot["eval_acc"].set_marker("o")`.

## Once per epoch

Log after the inner loop. x is the count reached, i.e. batches so far:

```python
plot = LivePlot("loss", "acc")
for epoch in range(epochs):
    for batch in plot(loader, desc=f"epoch {epoch}"):
        plot.log(loss=train_step(batch))
    plot.log(acc=evaluate())      # at x = (epoch + 1) * len(loader)
plot.finish()
```

## Reading values back

- `plot.data`: full history, `{metric: (xs, values)}`.
- `plot.latest`: the last value of each metric.

Both fill whether or not anything is drawn.

```python
xs, losses = plot.data["loss"]
best = min(losses)
```

## Collecting without drawing

`show=False` draws nothing; metrics are still collected and the bar still runs. `progress=False` turns the bar off. The two are independent.
