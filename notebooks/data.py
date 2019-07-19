import pandas as pd
import numpy as np
import pymc3 as pm

from sklearn import preprocessing


def load_kruschke():
    le = preprocessing.LabelEncoder()
    k_df = pd.read_csv('iq.csv', index_col=0)
    le.fit(k_df['treatment'])
    k_df['treatment_enc'] = le.transform(k_df['treatment'])
    return k_df


def load_ic50():
  chem_data = [(0.00080, 99), (0.00800, 91), (0.08000, 89), (0.40000, 89), (0.80000, 79), (1.60000, 61), (4.00000, 39), (8.00000, 25), (80.00000, 4)]
  chem_df = pd.DataFrame(chem_data)
  chem_df.columns = ['concentration', 'activity']
  chem_df['concentration_log'] = chem_df['concentration'].apply(
      lambda x : np.log10(x))
  return chem_df


def load_decay():
  tau = 71.9  # indium decay half life
  A = 42  # starting magnitude
  C = 21  # measurement error
  noise_scale = 1
  t = np.arange(0, 1000)

  def decay_func(ts, noise):
    return A * np.exp(-t / tau) + C + np.random.normal(0, noise, size=(len(t)))

  data = {'t': t, 'activity': decay_func(t, noise_scale)}
  df = pd.DataFrame(data)
  return df


def load_biofilm():
  le = preprocessing.LabelEncoder()
  df = pd.read_csv('biofilm.csv')
  le.fit(df['isolate'])
  df['isolate_enc'] = le.transform(df['isolate'])
  # Convert continuous columns to floatX for GPU compatibility.
  continuous_cols = [
      'OD600', 'ST', 'replicate', 'measurement', 'normalized_measurement'
      ]
  for c in continuous_cols:
      df[c] = pm.floatX(df[c])

  return df


def dose_response(x, lower, upper, slope, c50):
    """
    :param x: Array of x-values.
    :param lower: Lower-plateau value
    :param upper: Upper-plateau value
    :param slope: Slope of dose response curve
    :param c50: The midpoint value of the curve.
    """
    return lower + (upper - lower) / (1 + np.exp(slope * (c50 - x)))


def load_dose_response():
  x1 = np.vstack([np.linspace(-2, 2, 10)] * 5)
  x2 = np.vstack([np.linspace(-2, 2, 10)] * 5)
  x3 = np.vstack([np.linspace(-2, 2, 10)] * 5)

  def noise(x, sd):
    return np.random.normal(0, scale=sd, size=x.shape)

  y1 = dose_response(x1, lower=3, upper=6, slope=4, c50=0) + noise(x1, 0.5)
  y2 = dose_response(x2, lower=2.5, upper=8, slope=3, c50=1.5) + noise(x2, 0.5)
  y3 = dose_response(x3, lower=3.2, upper=3.4, slope=1.5, c50=0.0) + noise(x3, 0.5)

  df = pd.DataFrame({
    'concentration': np.concatenate([x1.flatten(), x2.flatten(), x3.flatten()]),
    'response': np.concatenate([y1.flatten(), y2.flatten(), y3.flatten()]),
    'molecule': (
      [0] * len(x1.flatten()) + [1] * len(x2.flatten()) + [2] * len(x3.flatten())
      )
    })

  return df
