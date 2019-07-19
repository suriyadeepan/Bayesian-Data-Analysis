A GP based auxiliary network that estimates uncertainty on prediction, without interacting with the prediction model.

- GP will be bad at prediction but it can capture uncertainty
- If a GP is bad at prediction, it hasn't modeled the data well
  - But we can establish a baseline uncertainty score
  - Anything more that will be the uncertainty for a new prediction
- Can we make use of the prediction network's internals as input to the GP?
  - Can we use the final layer of the prediction network as input to GP?
- Representation of input data
  - We need a flat $n$-d array