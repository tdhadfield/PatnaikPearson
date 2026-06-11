
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import cupy as cp
#import scikit-learn as sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import qr
import pandas as pd
import random

np.random.seed(42)
use_gpu = torch.cuda.is_available()

def say_hello(this_name="world"):
    return f"Hello, {this_name}!" 

def dot_product(array1, array2, verbose=False):
    """ Dot product of two numpy arrays"""
    if verbose:
        print("reached dot_product : dot product of two numpy arrays")
        print("array1.shape = ", array1.shape)
        print("array2.shape = ", array2.shape)
    return np.dot(array1, array2)

def get_valid_bert_base_token_ids(verbose=False):
  # 0 : [PAD], 100 : [UNK], 101 : [CLS], 102 : [SEP], 103 : [MASK]
  # all others between 0 and 998 are [unusedN] - discard these
  # then 999 - 30521 are all valid
  if verbose:
      print("get_valid_bert_base_token_ids")
      print("returns a list of valid bert-base token ids")
      print("0 : [PAD], 100 : [UNK], 101 : [CLS], 102 : [SEP], 103 : [MASK]")
      print("all others between 0 and 998 are [unusedN] - discard these")
      print("then 999 - 30521 are all valid")
      
  valid_token_ids = [0,100,101,102,103]
  for i in range(999, 30522):
    valid_token_ids.append(i)
    
  if verbose:
      print("type(valid_token_ids) = ", type(valid_token_ids))
      print("len(valid_token_ids) = ", len(valid_token_ids))
      
  return valid_token_ids

def get_invalid_bert_base_token_ids(verbose=False):
  if verbose:
      print("get_invalid_bert_base_token_ids")
      print("returns a list of invalid bert-base token ids")
      print("0 : [PAD], 100 : [UNK], 101 : [CLS], 102 : [SEP], 103 : [MASK] - these are valid ids")
      print("all others between 0 and 998 are [unusedN] - these are invalid")
      print("then 999 - 30521 are all valid")
      
  invalid_token_ids = []
  for i in range(1,100):
    invalid_token_ids.append(i)
  for i in range(104,999):
    invalid_token_ids.append(i)
    
  if verbose:
      print("type(invalid_token_ids) = ", type(invalid_token_ids))
      print("len(invalid_token_ids) = ", len(invalid_token_ids))
      
  return invalid_token_ids

def get_valid_invalid_bert_base_token_embeddings(bert_base_model, verbose=False):
  
  if verbose:
      print("get_valid_invalid_bert_base_token_embeddings")
      print("token embeddings from the bert-base model, corresponding to valid and invalid tokens")

  valid_token_ids = get_valid_bert_base_token_ids()

  # access token embeddings directly
  token_embedding_matrix = bert_base_model.embeddings.word_embeddings.weight
  total_num_tokens = token_embedding_matrix.shape[0]
  embedding_vector_size = token_embedding_matrix.shape[1]
  
  if verbose:
    print("total_num_tokens = ", total_num_tokens)
    print("embedding_vector_size = ", embedding_vector_size)

  # Convert the tensor into a numpy array using the .numpy () method. Convert that numpy array into a DataFrame using pd.DataFrame ()

  num_valid_tokens = len(valid_token_ids)
  num_invalid_tokens = total_num_tokens - num_valid_tokens
  embedding_vector_size=768

  valid_embeddings = np.zeros((num_valid_tokens, embedding_vector_size))
  invalid_embeddings = np.zeros((num_invalid_tokens, embedding_vector_size))

  valid_count=0
  invalid_count=0
  for i in range(total_num_tokens):
    if i in valid_token_ids:
      valid_embeddings[valid_count,:] = (token_embedding_matrix[i]).detach().cpu().numpy()
      valid_count += 1
    else:
      invalid_embeddings[invalid_count,:] = (token_embedding_matrix[i]).detach().cpu().numpy()
      invalid_count += 1
      
  if verbose:
      print("valid_embeddings.shape = ", valid_embeddings.shape)
      print("invalid_embeddings.shape = ", invalid_embeddings.shape)

  return valid_embeddings, invalid_embeddings
  
def generate_embedding_weights_summary_stats(these_embeddings, rescale_factor = 1.0):
    
  print("summary stats for embedding weights")
  print("type :", type(these_embeddings))
  print("shape :", these_embeddings.shape)
  print("min :", np.min(these_embeddings))
  print("max :", np.max(these_embeddings))
  print("mean :", np.mean(these_embeddings))
  print("std :", np.std(these_embeddings))

  custom_bins = []
  this_min = np.min(these_embeddings)
  this_max = np.max(these_embeddings)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  num_bins = 100
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  flattened_array = these_embeddings.flatten()
  print("flattened_array.shape = ",flattened_array.shape)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(flattened_array, bins=custom_bins)

  print("Histogram counts:", hist)

  # Plot histogram
  plt.hist(flattened_array, bins=custom_bins, edgecolor='black')
  plt.title("embedding_weights")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()
  return
  
def generate_embedding_norm_summary_stats(these_embeddings, rescale_factor = 1.0):
    
  # ** TO DO : adapt to GPU?
    
  print("summary stats for the norms of token embeddings")
  print("type :", type(these_embeddings))
  print("shape :", these_embeddings.shape)
  num_embeddings = these_embeddings.shape[0]
  print("num_embeddings :", num_embeddings)
  these_norms = np.linalg.norm(these_embeddings, axis=1)
  print("shape :", these_norms.shape)
  print("min :", np.min(these_norms))
  print("max :", np.max(these_norms))
  print("mean :", np.mean(these_norms))
  print("std :", np.std(these_norms))
    
  custom_bins = []
  this_min = np.min(these_norms)
  this_max = np.max(these_norms)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  num_bins = 100
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  flattened_array = these_norms.flatten()
  print("flattened_array.shape = ",flattened_array.shape)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(flattened_array, bins=custom_bins)

  print("Histogram counts:", hist)

  #import matplotlib.pyplot as plt

  # Plot histogram
  plt.hist(flattened_array, bins=custom_bins, edgecolor='black')
  plt.title("embedding_norms")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()
  return
  
def generate_norm_similarities(these_embeddings, num_random_indices=100, eps=0.0, rescale_factor = 1.0):
    
  # ** TO DO : adapt to GPU?
    
  print("generate_norm_similarities()")
  print("generate a number of random indices. take the corresponding embeddings, and rescale them to have norm 1. generate the symmetric matrix of dot products of these embeddings")  
      
  num_embeddings = these_embeddings.shape[0]
  
  print("num_embeddings :", num_embeddings)
  if num_random_indices > num_embeddings:
    num_random_indices = num_embeddings
  random_indices = np.random.choice(num_embeddings, num_random_indices, replace=False)

  random_embeddings = these_embeddings[random_indices]
  print("random_embeddings.shape = ",random_embeddings.shape)
  X = random_embeddings
  these_norms = np.linalg.norm(random_embeddings, axis=1)
  this_eps = np.ones(len(these_norms)) * eps
  these_norms = these_norms + this_eps
  one_over_norms = 1.0 / these_norms
  rescale_diag = np.diag(one_over_norms)
  print("rescale_diag.shape = ",rescale_diag.shape)
  X = np.matmul(rescale_diag, random_embeddings)
  XT = np.transpose(X)
  print("X.shape = ",X.shape)
  print("XT.shape = ",XT.shape)
  A = np.matmul(X, XT)
  print("A.shape = ",A.shape)
  n=5
  print("A = ",A[0:n,0:n])
  print("A.min :", np.min(A))
  print("A.max :", np.max(A))
  print("A.mean :", np.mean(A))
  print("A.std :", np.std(A))
  
  custom_bins = []
  this_min = np.min(A)
  this_max = np.max(A)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  num_bins = 100
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  flattened_array = A.flatten()
  print("flattened_array.shape = ",flattened_array.shape)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(flattened_array, bins=custom_bins)

  print("Histogram counts:", hist)

  # Plot histogram
  plt.hist(flattened_array, bins=custom_bins, edgecolor='black')
  plt.title("norm similarities")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()
  return
  
# normalise all vectors : first, demean; second, normalise
def normalise_vectors(all_vectors, demean=True, verbose=False):
    
  # ** TO DO : adapt to GPU?
    
  if verbose:
      print("normalise_vectors")
      print("normalise all vectors, one by one : first, demean (optional); second, rescale to norm 1")
      print("demean = ", demean)

  epsilon = 0.000001

  all_ones = np.ones(all_vectors.shape[1])
  final_vectors = np.zeros(all_vectors.shape)

  for i in np.arange(0, all_vectors.shape[0]):
    this_x = all_vectors[i,:]
    if demean:
        this_mean = np.mean(this_x)
        this_x = this_x - this_mean
    this_norm_sq = np.dot(this_x, this_x)
    this_norm = np.sqrt(this_norm_sq + epsilon)
    final_x = this_x * (1.0 / this_norm)

    final_mean = np.mean(final_x)
    final_norm_sq = np.dot(final_x, final_x)

    final_vectors[i,:] = final_x

  return final_vectors
  
def normalise_single_vector(this_vector, demean=True, verbose = False):
    
    # ** TO DO : adapt to GPU?

    if verbose:
        print("this_vector.shape = ", this_vector.shape)
        
    #convert to a 2-d array, the correct shape for normalisation:
    two_dim_array = np.zeros((1, this_vector.shape[0]))
    two_dim_array[0,:] = this_vector
    
    #now we can normalise it
    two_dim_array_normalised = normalise_vectors(two_dim_array, demean, verbose)

    #convert back to a 1-d array
    result = two_dim_array_normalised[0, :]
    if verbose:
        print("result.shape = ",result.shape)
        
    return result

def residualise(these_vectors, this_residualiser, normalise=False, demean=True, verbose=False):
    
    # ** TO DO : adapt to GPU?
    
    print("residualise : demean = ", demean)

    if verbose:
        print("residualise : first (optional) we normalise all the inputs. then we residualise. then we normalise the residuals")
        print("normalise = ",normalise)
        print("these_vectors.shape = ", these_vectors.shape)
        print("this_residualiser.shape = ", this_residualiser.shape)

    # first we normalise all the inputs
    X1 = these_vectors
    y1 = this_residualiser
    if normalise:
        X1 = normalise_vectors(these_vectors, demean, verbose=True)
        y1 = normalise_single_vector(this_residualiser, demean, verbose=True)

    #residualise X1 against y1
    X2 = np.zeros((X1.shape))
    n = 3
    if verbose:
        print(X2.shape)
        print(X2[0:n,0:n])

    for col in range(0, X1.shape[0]):
        this_embedding = X1[col,:]
        this_dot_product = np.dot(this_embedding, y1)
        this_residual = this_embedding - this_dot_product * y1
        if normalise:
            this_residual = normalise_single_vector(this_residual, demean, verbose)
        if verbose:
            print("this_embedding.shape = ",this_embedding.shape)
            print("this_dot_product =", this_dot_product)
            print("sanity check1 - this should be 0 : ", np.dot(this_residual, y1))
            print("sanity check2 - this should be 1 : ", np.dot(this_residual2[0,:], this_residual2[0,:]))
        X2[col,:] = this_residual

    return X2    


def sparsify(these_vectors, demean=True, verbose = False, similarity_threshold = 0.99):
    
  # ** TO DO : adapt to GPU?
    
  if verbose:
      print("take a (large) collection of vectors, and reduce it to a smaller collection by sparsification")
      print("this is done by iterating over the vectors one by one, calculating the dot product (cosine) similarity with each subsequent vector, and eliminating the subsequent vector if this dot product exceeds the similarity threshold")
      print("the end result is a collection of vectors for which any pair have dot product similarity less than the similarity threshold")
      print("this is a form of clustering")
      print("demean = ", demean)

  # input is a numpy array

  # normalise all vectors : first, demean; second, normalise
  normalised_vectors = normalise_vectors(these_vectors, demean, verbose)
  num_vectors = normalised_vectors.shape[0]
  embedding_dim = normalised_vectors.shape[1]

  valid_vectors = np.ones((num_vectors))
  for i in range(num_vectors):
    if valid_vectors[i] == 1:
      these_cosine_similarities = np.zeros((num_vectors))
      v_i = normalised_vectors[i,:]
      for j in range(i+1, num_vectors):
        if valid_vectors[j] == 1:
          v_j = normalised_vectors[j,:]
          dot_v_i_v_j = np.dot(v_i, v_j)
          if verbose:
            print(i, j, dot_v_i_v_j)
          these_cosine_similarities[j] = dot_v_i_v_j
          if dot_v_i_v_j > similarity_threshold:
            valid_vectors[j] = 0
            if verbose:
              print("eliminating ", j)

  num_valid = int(np.sum(valid_vectors))
  if verbose:
    print("final num_valid = ", num_valid)
  final_output = np.zeros((num_valid, embedding_dim))
  valid_count=0
  for i in range(num_vectors):
    if valid_vectors[i] == 1:
      final_output[valid_count,:] = normalised_vectors[i,:]
      valid_count += 1

  return final_output
  
# distance between two 1-dimensional numpy arrays
def get_euclidean_dist(x, y, verbose=False):
    
  # ** TO DO : adapt to GPU?
  
  if verbose:
      print("Euclidean distance between two 1-dimensional numpy arrays")
      print("x.shape = ", x.shape)
      print("y.shape = ", y.shape)
      
  sum_sq = 0.0
  for i in range(len(x)):
    xi = x[i]
    yi = y[i]
    sum_sq += (xi - yi)**2
  euclidean_dist = np.sqrt(sum_sq)
  
  if verbose:
      print("euclidean_dist = ",euclidean_dist)
      
  return euclidean_dist
  
  
def generate_distance_matrix(these_vectors, demean=True, verbose=False):
    
  # ** TO DO : adapt to GPU?
    
  if verbose:
      print("generate_distance_matrix : given a collection of vectors (data points), generate the symmetric matrix of pairwise euclidean distances")
      print("type(these_vectors) = ", type(these_vectors))
      print("these_vectors.shape = ", these_vectors.shape)
      print("demean = ", demean)

  data = normalise_vectors(these_vectors, demean, verbose)

  num_vectors = data.shape[0]
  if verbose:
    print("num_vectors = ", num_vectors)
  distance_matrix = np.zeros((num_vectors, num_vectors))

  for i in np.arange(0, num_vectors):
    for j in np.arange(i+1, num_vectors):
      vector_i = data[i,:]
      vector_j = data[j,:]
      d_ij = get_euclidean_dist(vector_i, vector_j, verbose)
      distance_matrix[i,j] = d_ij
      distance_matrix[j,i] = d_ij
      
  if verbose:
    print("type(distance_matrix) = ", type(distance_matrix))
    print("distance_matrix.shape = ", distance_matrix.shape)

  return distance_matrix
  
def generate_reduced_array(num_vectors, embedding_dim, dimension_reduction, verbose = False):
  reduced_dim = embedding_dim - dimension_reduction
  random_array = np.random.randn(num_vectors, reduced_dim)
  full_array = np.zeros((num_vectors, embedding_dim))
  full_array[:,0:k] = random_array
  
  if verbose:
      print("num_vectors = ", num_vectors)
      print("embedding_dim = ", embedding_dim)
      print("dimension_reduction = ", dimension_reduction)
      print("reduced_dim = ", reduced_dim)
      print("random_array.shape = ", random_array.shape)
      print("full_array.shape = ", full_array.shape)
  return full_array

def bert_base_inference(this_model, these_inputs, attention_mask, skip_all_embeddings=True, verbose=False):

  final_layer_index = 12
  embedding_dim = 768

  if verbose:
    print("type(these_inputs) =", type(these_inputs))
    print("these_inputs.shape =", these_inputs.shape)

  final_layer = np.zeros((embedding_dim, embedding_dim))
  # skip all embeddings (i.e., even the addition of positional and token type embeddings)
  if skip_all_embeddings:
    final_layer = bert_base_inference_skip_all_embeddings(this_model, these_inputs, attention_mask, verbose=verbose)
  else:
    final_layer = bert_base_inference_include_positional_embeddings(this_model, these_inputs, attention_mask, verbose=verbose)
    
  if verbose:
    print("type(final_layer) = ", type(final_layer))
    print("final_layer.shape = ", final_layer.shape)

  return final_layer 

def bert_inference_skip_all_embeddings(this_model, these_inputs, attention_mask, verbose=False):

  final_layer_index = 12
  embedding_dim = 768

  #expects these_inputs.shape == (1, 1, 768)

  if verbose:
    print("type(these_inputs) =", type(these_inputs))
    print("*** Expected these_inputs.shape is (1, 1, 768) ***")
    print("these_inputs.shape =", these_inputs.shape)

  # Feed directly into encoder
  outputs = this_model.encoder(
      hidden_states=these_inputs,
      attention_mask=this_model.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device),
      output_hidden_states=True
  )

  #get the final layer
  final_layer = (outputs.hidden_states[final_layer_index].detach().numpy())[0]
  if verbose:
    print("type(final_layer) = ", type(final_layer))
    print("final_layer.shape = ", final_layer.shape)

  return final_layer  
    
def bert_base_inference_include_positional_embeddings(this_model, these_inputs, attention_mask, verbose=False):

  if verbose:
    print("type(these_inputs) =", type(these_inputs))
    print("*** Expected these_inputs.shape is (1, 1, 768) ***")
    print("these_inputs.shape =", these_inputs.shape)

  embedding_dim = these_inputs.shape[2]
  final_layer = np.zeros((embedding_dim, embedding_dim))

  print_this = True
  for i in np.arange(0, these_inputs.shape[1]):
    this_input_vector = these_inputs[0,i,:]
    this_input_vector = torch.unsqueeze(torch.unsqueeze(this_input_vector, 0),0)

    if verbose and print_this:
      print("this_input_vector.shape =", this_input_vector.shape)

    # Pass directly as inputs_embeds instead of input_ids:
    output_vector = this_model(inputs_embeds=this_input_vector)

    if verbose and print_this:
      print("type(output_vector) =", type(output_vector))
      print("output_vector.keys() =", output_vector.keys())

    last_hidden_state = output_vector.last_hidden_state
    final_vector = last_hidden_state.detach().numpy()

    if verbose and print_this:
      print("type(final_vector) =", type(final_vector))
      print("final_vector.shape =", final_vector.shape)

    final_layer[i,:] = final_vector
    print_this = False

  if verbose:
    print("type(final_layer) = ", type(final_layer))
    print("final_layer.shape = ", final_layer.shape)

  return final_layer
  
def get_average_valid_bert_base_embedding():
    
   # ** TO DO : deprecate

    avg = np.array([ 1.13084192e-02, -1.72215811e-02, -1.41714168e-02, -2.74185247e-02,
   2.98537331e-02, -6.42583949e-03, -2.42618347e-02,  4.16709138e-03,
   1.11692373e-02, -4.06155263e-02, -1.35838335e-02, -4.83584226e-02,
  -4.13017340e-02,  4.96121871e-02,  1.78355660e-02, -2.54912581e-02,
  -4.69590329e-02, -2.86391754e-02,  3.21339747e-02, -2.11959472e-02,
  -1.91453447e-02, -5.28497078e-03,  8.82985522e-03,  3.20782928e-03,
  -7.90428483e-02, -1.04079559e-02,  2.89219318e-02, -2.14861649e-02,
   1.48650061e-02, -7.79963590e-03,  8.97508906e-03,  2.10882902e-03,
   1.27321133e-02,  1.05645595e-02, -1.69112090e-02, -2.38637766e-02,
  -1.90533016e-02,  6.47673533e-03, -4.59222775e-02,  2.71221729e-02,
  -5.80885480e-03,  9.14678611e-03, -2.43531236e-02, -3.71557564e-03,
   1.48051832e-02,  1.66223456e-02,  1.12191612e-02, -4.77844003e-03,
   1.33904535e-02,  3.05887474e-04,  1.65432077e-03,  1.35902316e-02,
  -1.24025803e-02,  9.85183668e-02, -9.14891399e-03, -1.93592604e-02,
   9.02638890e-03,  1.20396629e-02,  2.25755599e-02,  1.90945075e-02,
   2.45243878e-02, -5.66522124e-03,  1.19119505e-01, -1.64141835e-02,
  -1.54688070e-02,  7.08495663e-03,  6.56908770e-04, -7.46583735e-03,
   1.69385693e-02,  1.57647053e-03,  1.03968233e-02,  2.06339265e-02,
  -2.58182810e-02, -5.73670966e-02, -5.96087810e-02,  4.26947130e-03,
  -3.76456495e-02,  1.35936756e-01,  6.93226341e-03, -2.36239310e-02,
  -1.19383005e-02,  1.71738215e-03,  2.07884285e-02,  1.39557949e-03,
  -1.57755476e-02,  1.31444919e-02, -4.76048824e-04, -1.07611332e-02,
   1.47663022e-02,  6.62903819e-03, -1.68732963e-02,  2.29713184e-02,
  -6.29975820e-02,  1.54610835e-02, -9.47935873e-03, -2.96283437e-02,
   1.31583597e-01, -2.27532003e-02, -2.52087869e-02, -7.78397134e-03,
   1.41448308e-02,  1.37490542e-02,  2.53899756e-02, -2.62883032e-02,
  -2.73764523e-02,  1.07645406e-01, -2.64147367e-02,  2.43184866e-02,
   1.69962779e-02,  2.89334683e-03, -3.19371274e-02,  1.23076384e-02,
  -5.88722176e-03, -1.19382138e-02,  5.05342671e-03,  2.52756474e-02,
   1.15857666e-02, -8.93152515e-03, -1.94648338e-02, -6.05392748e-04,
   1.48152175e-02,  1.86579433e-01, -5.30818109e-03, -6.08050785e-03,
  -1.21455736e-01, -1.22979730e-02,  1.92017612e-03, -1.43900416e-01,
  -1.27299066e-02, -4.49465943e-02,  2.28975954e-02, -2.30340820e-03,
   1.48777500e-01,  2.40598581e-02, -2.01261666e-02,  6.20954207e-03,
   5.78189319e-03,  4.97904665e-03,  2.18188856e-02,  1.33813591e-01,
   1.44866232e-03,  6.29061666e-03, -1.59401067e-03,  2.26942542e-02,
   3.57316109e-03, -2.38938674e-02,  2.02266285e-03, -2.07379550e-02,
  -3.18395287e-02, -1.75796580e-02,  2.62113322e-03, -2.39017762e-02,
  -1.15677848e-01,  2.22292025e-03, -7.86365741e-03, -2.31914252e-02,
   8.85865959e-03, -4.48986497e-03, -1.21662425e-03,  1.34148046e-01,
   5.53733913e-03,  8.07666707e-03, -9.22575516e-03, -2.42707226e-02,
  -7.26197050e-03, -1.55745419e-02,  2.23384758e-02,  8.83802497e-03,
   1.47529825e-01, -1.28332712e-03, -1.31124735e-02,  3.47243678e-02,
   2.77009533e-02, -7.85741628e-03,  1.08840159e-02,  1.73207677e-01,
  -3.42812810e-04, -4.40130101e-02,  1.58917485e-02,  2.65065882e-02,
   1.91654604e-02, -8.85565053e-03, -6.79010560e-02, -2.04931736e-03,
   2.70146879e-02, -1.45261211e-02,  1.00779099e-02, -1.88655224e-02,
   4.04940348e-03,  1.48999166e-02,  5.83193224e-03,  6.87562316e-04,
  -6.43779380e-02, -2.12560876e-02, -3.07150850e-02, -2.36318578e-02,
  -1.47264003e-02, -1.53675669e-02, -5.26947449e-02, -3.55137275e-02,
  -3.49320190e-02,  2.41346467e-02,  1.14669149e-01, -3.72137983e-02,
  -9.43411453e-03,  1.72035774e-02, -3.04993040e-04, -9.86743433e-03,
  -1.44058030e-02,  7.43421707e-03,  6.82401112e-03,  8.14463308e-03,
  -9.19321245e-03, -3.48007366e-03,  1.88404261e-02,  6.78054943e-02,
  -8.01400362e-02,  1.99915505e-02,  9.79112103e-02,  2.75038385e-03,
   1.48036032e-02,  3.64266833e-03, -8.34633470e-03,  9.64227834e-03,
  -3.87703806e-02,  4.62200761e-02, -2.84352675e-03, -3.36092841e-02,
   3.34421148e-03, -3.24685089e-02, -6.74195737e-03, -1.12494710e-02,
  -5.80593166e-02, -4.57696518e-03, -6.36531986e-03,  7.62200235e-03,
   7.44952491e-03,  4.03390803e-03,  4.92907026e-03, -9.83396661e-03,
  -3.61347611e-02,  2.07590925e-03, -1.00556758e-02, -4.36313631e-02,
  -3.36845191e-02,  1.39888025e-01,  4.14170685e-03, -1.95593746e-02,
   6.57805635e-03, -8.24051234e-02, -4.68260091e-02,  1.13772074e-02,
   4.74968518e-03,  3.42687589e-02,  2.84384309e-02,  2.84249907e-02,
  -1.46052436e-03, -2.33200013e-02,  2.39918889e-02,  6.82978355e-03,
   1.03306257e-02, -1.91522908e-02,  3.13359458e-02, -2.23641437e-03,
  -6.03414875e-02, -9.38153173e-03,  6.90087958e-03, -1.30347132e-02,
  -1.89753492e-02, -3.01169711e-03, -3.59374458e-02,  7.15668423e-03,
   2.12879692e-02, -1.48647003e-02,  3.80434164e-03,  1.11996425e-02,
   8.69286808e-03, -2.62625158e-02,  4.13912795e-03, -8.53629785e-03,
  -3.82918688e-03,  3.63532850e-03,  6.35887007e-03,  7.98717926e-02,
   1.93625364e-02, -5.81506209e-02, -1.45591361e-02, -8.34907711e-03,
  -1.41811024e-02,  9.18216034e-03, -5.83685576e-02,  1.18148056e-01,
   2.52187585e-02, -4.14077683e-03, -2.07613722e-02,  1.92065113e-03,
   2.84277556e-02, -2.48745693e-02, -5.42969793e-02, -7.24178309e-03,
  -2.54016272e-03, -2.19770690e-02,  6.65240770e-03,  7.70726214e-03,
  -5.38489896e-02,  1.86236876e-02,  4.19111310e-03,  1.78886185e-03,
   9.33150624e-02, -2.08852276e-02, -3.35598129e-03,  1.22440200e-01,
   8.14346063e-03, -3.94416106e-02, -2.84984378e-02, -4.32410859e-02,
  -2.85596679e-02,  5.87281471e-03,  1.72281266e-02,  1.25368857e-01,
  -1.10702890e-03,  1.95367991e-02, -4.65158917e-03, -9.04263128e-03,
  -2.31242418e-02,  2.51503526e-02,  4.27960635e-03, -1.08681331e-02,
   1.26101112e-02,  2.97725842e-02, -2.14622537e-02,  7.97195622e-02,
  -4.87821367e-02,  1.18895720e-03,  1.82514359e-03,  1.81334294e-02,
   2.71015316e-02, -5.07083246e-03,  6.00653037e-03,  7.88284216e-03,
  -1.38496621e-02, -1.60665617e-03, -5.56725323e-03, -4.06533196e-03,
   3.61437651e-02,  1.44535666e-01, -1.13321513e-03, -4.80371903e-02,
  -1.54109438e-02,  1.47862545e-02, -3.28612968e-02, -2.22059724e-03,
  -1.10012346e-02,  1.07238632e-01, -2.64958054e-03, -1.80304124e-02,
  -1.08565291e-02, -5.26901476e-04, -7.35414959e-03,  1.70699571e-02,
   1.16973725e-02, -1.95270944e-02,  1.60017872e-02, -1.72626162e-02,
   2.73187312e-02, -1.44896180e-02, -8.35385494e-04, -2.48563761e-03,
  -3.29183884e-02, -4.30411625e-02,  1.91999083e-02, -2.61639741e-02,
   2.55387042e-02, -2.01963449e-02, -4.20250260e-02, -2.28268864e-02,
  -3.40141034e-03, -5.56466190e-03, -2.04078761e-02,  2.16355732e-02,
   9.88664609e-03, -4.59912831e-02, -9.09282829e-03,  8.09964749e-03,
  -1.05258829e-02, -1.21176510e-02,  8.14855714e-04, -3.82739976e-03,
   8.88786884e-03,  2.25225885e-03, -5.41708587e-02, -6.15615965e-03,
  -2.68711580e-02, -2.68935300e-02, -2.37893801e-03,  3.66466762e-03,
  -4.66136265e-03,  8.62187000e-03, -5.27641211e-02, -1.15001066e-03,
   2.47376825e-02, -1.84124132e-01, -1.37513960e-02, -2.79176528e-02,
  -8.21169738e-03,  1.29806030e-02, -6.23219199e-03,  1.36354106e-02,
   4.08941068e-03, -9.50222885e-03, -1.94446162e-02, -2.66915554e-02,
  -1.36220087e-03, -3.93264020e-02,  8.23023660e-03,  1.87253404e-02,
  -3.92331759e-02, -3.18086649e-02,  1.25493774e-02, -3.02616716e-02,
   2.14761077e-02,  1.53498543e-03, -2.65592536e-03, -4.36347912e-02,
  -2.92347459e-02, -9.40934056e-03,  2.04763104e-03, -4.46949998e-02,
  -8.86850266e-03, -9.16852745e-02, -7.00622099e-04, -1.26634540e-02,
  -2.23699753e-02,  3.04991622e-02,  1.84480259e-02, -1.64857532e-03,
   1.80364415e-02, -6.70237599e-03,  1.44241450e-02,  2.17502238e-02,
   4.05697769e-03,  3.09802775e-02, -3.14136728e-02,  1.59632322e-01,
  -6.39782989e-03, -1.62788550e-02, -5.99170202e-03,  7.17035110e-03,
  -1.55791220e-02, -3.60507236e-02,  1.58713524e-02, -5.05494910e-04,
   5.67770975e-03,  3.44094672e-03,  9.62532192e-03, -7.59137536e-03,
  -1.59083721e-02,  4.33359560e-03, -8.93658438e-03, -7.84320378e-03,
  -2.13411660e-02,  6.29466841e-02, -1.49833348e-02,  1.31653577e-02,
  -1.21752849e-02, -2.29615449e-02, -5.37091699e-03,  4.39063660e-03,
   2.22605307e-02, -1.53808392e-02,  1.65603799e-02, -5.29354449e-02,
  -3.90296771e-02,  1.92766940e-02, -1.77871499e-02,  2.13868063e-02,
   1.26705454e-02, -8.42040462e-03, -8.91867606e-04,  7.67267817e-03,
  -8.72522042e-03,  8.66671071e-03,  1.53082249e-01,  1.88811716e-02,
  -5.95102986e-04,  9.20117035e-03,  1.13164580e-02, -1.48529880e-02,
   8.63022674e-03, -6.44731507e-03, -1.11637995e-02, -2.57899369e-02,
  -1.48755419e-02, -1.62477317e-02,  1.66541323e-03,  1.49332885e-02,
  -2.40177171e-02,  1.24913695e-02, -3.65861063e-02, -4.94657129e-02,
   1.37476135e-02, -2.12339151e-02, -2.57349303e-02, -3.70115603e-02,
  -4.08293849e-02, -5.36773596e-02, -2.77653910e-02,  3.39735731e-03,
   3.92158720e-03, -2.19063734e-02, -2.15320767e-02, -5.80495920e-02,
  -1.57634955e-03, -2.33745467e-02, -7.22740192e-03, -1.67754684e-02,
   8.68213230e-03,  1.40494013e-01,  2.68511318e-03, -1.77644492e-03,
  -1.36589065e-02, -3.71410203e-03,  1.95884888e-02,  2.01261082e-02,
   7.15569120e-03,  1.48273581e-02, -1.68393947e-02, -2.32106830e-02,
  -9.71574435e-03,  1.80932985e-01, -2.58467226e-02, -4.08072335e-02,
   2.18091977e-03,  7.97981378e-03,  3.55129222e-04,  7.25916687e-03,
  -1.96398432e-02, -6.28331866e-03, -1.96943417e-02,  1.07044422e-02,
  -1.88097191e-02,  9.72540442e-02,  1.68266358e-02, -1.39970174e-02,
   1.77522541e-02,  1.21453501e-01,  2.25818699e-02, -1.85191211e-02,
  -3.53291734e-02, -1.43756335e-03, -4.62032938e-03,  1.04969762e-02,
  -1.27210193e-02,  1.89018941e-02,  8.63639985e-03, -1.45633958e-02,
   9.14691106e-03,  1.74686366e-02, -2.38596857e-02, -5.95819919e-03,
  -2.16259988e-03,  6.83748356e-03,  1.60892612e-02, -1.95489011e-02,
   8.11088686e-03, -2.53830899e-04, -2.40142553e-02, -1.22094890e-02,
  -3.46365601e-02,  2.45377263e-02,  8.15083336e-04, -1.06298882e-02,
   7.59711697e-03,  8.75694610e-03,  5.68790843e-03,  1.11862141e-02,
  -7.31989351e-03,  1.63859774e-01,  1.85581306e-02, -3.74518348e-03,
   5.72096506e-03, -2.35866197e-02, -1.51001094e-02,  2.56276778e-02,
   1.43283055e-02,  1.84685721e-02, -7.91525742e-03, -1.75199006e-02,
  -3.36527764e-02, -3.97695768e-02, -1.12098616e-02, -3.41075410e-02,
   2.09497140e-03, -2.85307328e-02,  1.15062412e-02,  2.31628530e-02,
  -7.49949168e-03,  8.87283046e-03, -1.69309962e-02,  6.29385533e-03,
  -3.32423809e-03,  2.07357743e-02, -3.78965153e-03,  3.47384668e-02,
   2.30298337e-02, -1.38294000e-02,  6.83982278e-03, -4.61059417e-02,
  -2.53906286e-02,  8.26466535e-03, -2.38701595e-02, -5.52104286e-02,
   1.65068026e-02,  1.08160540e-02,  1.32763290e-01, -2.58933439e-02,
   5.86485010e-03,  6.50582463e-03, -3.37666712e-02,  3.26233407e-02,
  -9.36446161e-03, -6.29968474e-04,  8.34167010e-03, -1.76742113e-02,
   1.26416550e-03, -4.21402447e-03, -3.88438368e-03,  2.75489887e-03,
   1.07772869e-02,  2.56493305e-03, -1.96697564e-02, -1.20862405e-02,
   7.69511430e-03, -1.90504057e-02,  1.60816160e-02,  4.48083487e-03,
   2.65545305e-02, -4.10696925e-03,  5.19012849e-03, -3.04567905e-02,
   7.94948602e-03, -2.69632198e-02, -2.31911242e-02,  1.55792739e-02,
   2.51191243e-02,  1.25706062e-02, -1.14414584e-02, -1.03035374e-02,
   6.34582050e-03,  2.29466114e-02, -1.32081705e-03, -3.04694986e-03,
   2.57349829e-03, -4.38728604e-02,  1.62110904e-01,  1.82218291e-02,
  -7.46688598e-03, -6.64960297e-03,  1.25270895e-02, -2.81180407e-02,
  -1.87040499e-02, -1.55435694e-02, -3.54653411e-03, -5.41688459e-03,
  -1.51739038e-02, -2.29021912e-02,  2.44296945e-02, -2.60920604e-02,
   1.05749264e-02,  3.12978006e-03,  1.37156769e-01, -1.09662462e-02,
  -3.23240823e-02,  5.38053400e-03, -6.39382657e-03,  8.84680917e-03,
  -1.68019708e-02, -6.67246349e-03, -2.73754411e-03,  4.12695784e-03,
   1.82784715e-02, -1.91458326e-03, -3.07742607e-02,  1.65856787e-02,
  -4.80900591e-03, -2.10456509e-03,  2.86911831e-03, -1.77132534e-05,
  -1.08823546e-02,  2.19189594e-02, -4.38726783e-03,  1.70914036e-02,
  -6.36450128e-03,  1.13847633e-02, -2.92361824e-03, -1.04362735e-02,
   1.75042601e-02,  1.80667534e-03, -1.22676322e-02,  3.75009545e-02,
   1.55577212e-02,  1.94966448e-02, -2.44882544e-02, -4.58122105e-02,
  -3.83748776e-02, -1.39479768e-02,  6.38407956e-03,  1.91697820e-02,
  -1.27283049e-02, -2.89821020e-02,  2.76443058e-02, -1.28467913e-02,
  -2.80675547e-03, -1.25590702e-02,  2.52126160e-02,  3.79752890e-03,
  -8.67725363e-03,  1.86887524e-02, -4.82170804e-03,  2.15036883e-02,
  -1.82837879e-02,  6.72212849e-03,  4.94004181e-03, -1.67669427e-02,
  -1.40186859e-03, -3.36212152e-03,  8.83205705e-03, -7.04254417e-03,
  -2.94295264e-02, -8.18000532e-04, -2.35360920e-02,  1.28985639e-02,
   5.17536762e-03, -1.67288707e-02, -5.03910822e-02,  2.32826484e-02,
   2.71808287e-02,  2.42017796e-02, -2.04954474e-02,  1.27501562e-02,
  -1.70181905e-02, -4.01029715e-03,  1.46318926e-02, -3.05424594e-02,
  -1.54742951e-03,  7.99676161e-03,  1.53432417e-02,  7.26708315e-03,
   1.11087819e-02,  1.74719075e-02,  3.96518610e-03, -2.35782816e-02,
   2.00478244e-02, -1.65583683e-02, -3.63933530e-02,  1.62410850e-02,
  -4.28085148e-02,  3.03947339e-03, -2.25559874e-02, -2.52840692e-03,
   1.04531594e-02, -1.16248872e-02, -2.26573382e-02, -1.26748001e-02,
   3.03898487e-03, -2.32054756e-03, -1.15721960e-02,  1.20602132e-03])

    return avg
    

def generate_normalised_random_vectors_with_one_common_component(num_vectors, embedding_dim, desired_correlation, demean=True, verbose=False):
    
    # ** TO DO : adapt to GPU?
    # generate the initial random embeddings
    random_embeddings_0 = np.random.randn(num_vectors, embedding_dim)

    # set the zero component of each vector to zero
    for i in range(0, num_vectors):
        random_embeddings_0[i,0] = 0.0

    n=3
    if verbose:
        print(random_embeddings_0[0:n,0:n])

    # normalise
    random_embeddings_1 = normalise_vectors(random_embeddings_0, demean, verbose)

    # define alpha
    alpha = math.sqrt(desired_correlation / (1.0 - desired_correlation))
    if verbose:
        print("alpha = ",alpha)

    # set the zero component to alpha
    for i in range(0, num_vectors):
        random_embeddings_1[i,0] = alpha

    # normalise again 
    random_embeddings_2 = normalise_vectors(random_embeddings_1, demean, verbose)
    if verbose:
        print(random_embeddings_2[0:n,0:n])

    return random_embeddings_2
    
def demean_embeddings(these_embeddings, verbose=True):
    
    # ** TO DO : adapt to GPU?
    these_embeddings_mean = np.mean(these_embeddings)
    if verbose:
        print("these_embeddings_mean =", these_embeddings_mean)

    these_embeddings_demeaned = these_embeddings - these_embeddings_mean * np.ones(these_embeddings.shape)

    return these_embeddings_demeaned
    
def generate_per_vector_summary_stats(these_embeddings, rescale_factor=1.0):

    num_vectors = these_embeddings.shape[0]
    print("generate_per_vector_summary_stats")
    print("these_embeddings.shape = ", these_embeddings.shape)
    print("num_vectors = ", num_vectors)

    these_means = np.zeros(num_vectors)
    these_stds = np.zeros(num_vectors)
    for col in range(0, num_vectors):
        this_vector = these_embeddings[col,:]
        this_mean = np.mean(this_vector)
        these_means[col]=this_mean
        this_std = np.std(this_vector)
        these_stds[col]=this_std

    this_dict = {"means" : these_means, "stds" : these_stds}

    for kk in this_dict:
        print(kk)
        this_array = this_dict[kk]
        this_min = np.min(this_array)
        this_max = np.max(this_array)

        print("summary stats for per-vector", kk)
        print("min :", this_min)
        print("max :", this_max)
        print("mean :", np.mean(this_array))
        print("std :", np.std(this_array))

        custom_bins = []
        this_mid = 0.5 * (this_min + this_max)
        min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
        max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
        num_bins = min(100, max(1, 10 * int(num_vectors / 100)))
        step_size = (max_value - min_value) / num_bins
        for i in range(num_bins + 1):
            custom_bins.append(min_value + i * step_size)

        # Compute histogram with custom bins
        hist, bin_edges = np.histogram(this_array, bins=custom_bins)

        # Plot histogram
        plt.hist(this_array, bins=custom_bins, edgecolor='black')
        plt.title(kk)
        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.show()

    # Using NumPy
    pearson_corr_np = np.corrcoef(these_means, these_stds)[0, 1]
    print(f'Pearson correlation using NumPy: {pearson_corr_np}')

    x_values = these_means
    y_values = these_stds
    plt.scatter(x_values, y_values, color='blue', marker='o', label='std vs mean')

    # Add labels and title
    plt.xlabel('mean')
    plt.ylabel('std')
    #plt.title('Scatter Plot with Two Data Categories')

    # Show legend
    plt.legend()

    # Optional: Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Display the plot
    plt.show()
    
    return
    
def generate_random_embeddings_with_given_characteristics(num_vectors, embedding_dim, mean_mean, mean_std, std_mean, std_std, mean_std_correlation, verbose=False, sort_prop=0.0):

    this_shape = (num_vectors, embedding_dim)
    random_embeddings = np.zeros(this_shape)
    if verbose:
        print("random_embeddings.shape = ", random_embeddings.shape)

    #generate a 1-d array of means
    these_means = np.random.normal(loc=mean_mean, scale=mean_std, size=num_vectors)
    if verbose:
        print("these_means.shape = ", these_means.shape)
        print("these_means mean = ", np.mean(these_means))
        print("these_means std = ", np.std(these_means))

    # Y = beta * X + Theta
    # X = mean
    # Y = std
    # Theta = random component of Y, uncorrelated with X
    # E(X) = mean_mean
    # Var(X) = mean_std ^ 2
    # E(Y) = std_mean
    # Var(Y) = std_std ^ 2
    # rho = corr(X,Y) = beta * (X_std / Y_std) = (beta * mean_std) / std_std
    # hence beta = rho * std_std / mean_std
    # Var(Theta) = (1 - rho^2) Var(Y) = (1 - rho ^ 2) std_std ^ 2
    # E(Theta) = E(Y) - beta * E(X) = std_mean - beta * mean_mean
    # Theta = Theta_mean + Theta_std * Z, where Z is N(0,1)

    beta = mean_std_correlation *  std_std / mean_std
    theta_mean = std_mean - beta * mean_mean
    theta_std = math.sqrt((1 - (mean_std_correlation * mean_std_correlation)) * std_std * std_std)
    
    these_stds = np.zeros(num_vectors)
    these_stds = beta * these_means + np.random.normal(loc = theta_mean, scale = theta_std, size = num_vectors)
    
    if verbose:
        print("these_stds.shape = ", these_stds.shape)
        print("these_stds mean = ", np.mean(these_stds))
        print("these_stds std = ", np.std(these_stds))

    max_sort_threshold = int(sort_prop * num_vectors)
    
    if verbose:
        print("sort_prop = ", sort_prop)
        print("max_sort_threshold = ", max_sort_threshold)
        
    for i in range(0, num_vectors):
        this_mean = these_means[i]
        this_std = these_stds[i]
        if this_std < 0.0:
            this_std = 0.0
        this_vector = np.random.normal(loc = this_mean, scale = this_std, size = embedding_dim)
        if i < max_sort_threshold:
            this_vector = np.sort(this_vector, axis=0)
        random_embeddings[i,:] = this_vector 

    if verbose:
        print("random_embeddings mean = ", np.mean(random_embeddings))
        print("random_embeddings std = ", np.std(random_embeddings))

    return random_embeddings
    
def display_stats(this_array, calc_norm=True):
    print("shape :", this_array.shape)
    print("min :", np.min(this_array))
    print("max :", np.max(this_array))
    print("sum :", np.sum(this_array))
    print("mean :", np.mean(this_array))
    print("std :", np.std(this_array))
    if calc_norm and len(this_array.shape) == 1:
        print("norm :", math.sqrt(np.dot(this_array, this_array)))
    return

def plot_histogram(this_array, rescale_factor=1.0, x_label = "x_label", y_label = "y_label", title = "this is the title", verbose=False):
    custom_bins = []
    this_min = np.min(this_array)
    this_max = np.max(this_array)
    this_mid = 0.5 * (this_min + this_max)
    min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
    max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
    num_bins = 100
    step_size = (max_value - min_value) / num_bins
    for i in range(num_bins + 1):
        custom_bins.append(min_value + i * step_size)
    
    # Compute histogram with custom bins
    hist, bin_edges = np.histogram(this_array, bins=custom_bins)

    if verbose:
        print("Histogram counts:", hist)

    plt.hist(this_array, bins=custom_bins, edgecolor='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    
# apply log to an array by vectorisation
def this_log(x):
  eps=1e-6
  return math.log(max(x,eps))

# Apply the function using np.vectorize
this_vec_log = np.vectorize(this_log)

def calculate_nu_alpha_W_WTW_square_W(d : int, 
                                      alpha : float, 
                                      verbose : bool = False
                                      ) -> tuple:
								 
  # d is the dimension of the ambient space
  # W is an d * d matrix
  # WTW is a d * d matrix
  
  # ** TO DO : use new alpha
  # ** TO DO : adapt to GPU?


  W = generate_square_weight_matrix(d, alpha)
  dim_W = W.shape[1]
  actual_nu_W_over_dim_W = calculate_PatnaikPearson_dim(W) / dim_W
  actual_alpha_W = calculate_alpha_given_nu_over_d_and_d(actual_nu_W_over_dim_W, dim_W)
  if verbose:
    print("alpha = ", alpha, ", actual_alpha_W = ", actual_alpha_W, ", actual_nu_W_over_dim_W = ", actual_nu_W_over_dim_W)

  WTW = W.T @ W
  dim_WTW = WTW.shape[1]
  actual_nu_WTW_over_dim_WTW = calculate_PatnaikPearson_dim(WTW) / dim_WTW
  estimate_nu_WTW_over_dim_WTW = estimate_product_nu_over_d(actual_nu_W_over_dim_W, actual_nu_W_over_dim_W)
  actual_alpha_WTW = calculate_alpha_given_nu_over_d_and_d(actual_nu_WTW_over_dim_WTW, dim_WTW) 
  estimate_alpha_WTW = estimate_product_alpha(actual_alpha_W, dim_W, actual_alpha_W, dim_W)
  if verbose:
    print("dim_WTW = ", dim_WTW)
    print("actual_nu_WTW_over_dim_WTW = ", actual_nu_WTW_over_dim_WTW, "estimate_nu_WTW_over_dim_WTW = ", estimate_nu_WTW_over_dim_WTW)
    print("actual_alpha_WTW = ", actual_alpha_WTW, "estimate_alpha_WTW = ", estimate_alpha_WTW)

  results_dict = { 
    "actual_nu_W_over_dim_W" : actual_nu_W_over_dim_W,
    "actual_alpha_W" : actual_alpha_W,
    "actual_nu_WTW_over_dim_WTW" :  actual_nu_WTW_over_dim_WTW,
    "estimate_nu_WTW_over_dim_WTW" :  estimate_nu_WTW_over_dim_WTW,
    "actual_alpha_WTW" :  actual_alpha_WTW,
    "estimate_alpha_WTW" : estimate_alpha_WTW
    }
   
  return results_dict
  
def calculate_nu_alpha_X_XTX_XXT(N : int, 
                                 d : int, 
                                 alpha : float, 
                                 verbose : bool = False
                                 ) -> tuple:

  # N is the number of data points
  # d is the dimension of the ambient space
  # X is an N * d matrix
  # XTX is a d * d matrix
  # XXT is an N * N matrix
  
  # ** TO DO : use new alpha
  # ** TO DO : adapt to GPU?


  X = generate_data_manifold(N, d, alpha)
  dim_X = X.shape[1]
  actual_nu_X_over_dim_X = calculate_PatnaikPearson_dim(X) / dim_X
  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(actual_nu_X_over_dim_X, dim_X)
  
  if verbose:
    print("alpha = ", alpha, ", actual_alpha_X = ", actual_alpha_X, ", actual_nu_X_over_dim_X = ", actual_nu_X_over_dim_X)

  XTX = X.T @ X
  dim_XTX = XTX.shape[1]
  actual_nu_XTX_over_dim_XTX = calculate_PatnaikPearson_dim(XTX) / dim_XTX
  estimate_nu_XTX_over_dim_XTX = estimate_product_nu_over_d(actual_nu_X_over_dim_X, actual_nu_X_over_dim_X)
  actual_alpha_XTX = calculate_alpha_given_nu_over_d_and_d(actual_nu_XTX_over_dim_XTX, dim_XTX) 
  estimate_alpha_XTX = estimate_product_alpha(actual_alpha_X, dim_X, actual_alpha_X, dim_X)
  
  if verbose:
    print("dim_XTX = ", dim_XTX)
    print("actual_nu_XTX_over_dim_XTX = ", actual_nu_XTX_over_dim_XTX, "estimate_nu_XTX_over_dim_XTX = ", estimate_nu_XTX_over_dim_XTX)
    print("actual_alpha_XTX = ", actual_alpha_XTX, "estimate_alpha_XTX = ", estimate_alpha_XTX)

  XXT = X @ X.T
  dim_XXT = XXT.shape[1]
  actual_nu_XXT_over_dim_XXT = calculate_PatnaikPearson_dim(XXT) / dim_XXT
  estimate_nu_XXT_over_dim_XXT = estimate_product_nu_over_d(actual_nu_X_over_dim_X, actual_nu_X_over_dim_X)
  actual_alpha_XXT = calculate_alpha_given_nu_over_d_and_d(actual_nu_XXT_over_dim_XXT, dim_XXT) 
  estimate_alpha_XXT = estimate_product_alpha(actual_alpha_X, dim_X, actual_alpha_X, dim_X)
  
  if verbose:
    print("dim_XXT = ", dim_XXT)
    print("actual_nu_XXT_over_dim_XXT = ", actual_nu_XXT_over_dim_XXT, "estimate_nu_XXT_over_dim_XXT = ", estimate_nu_XXT_over_dim_XXT)
    print("actual_alpha_XXT = ", actual_alpha_XXT, "estimate_alpha_XXT = ", estimate_alpha_XXT)

  results_dict = { 
    "actual_nu_X_over_dim_X" :  actual_nu_X_over_dim_X,
    "actual_alpha_X" : actual_alpha_X,
    "actual_nu_XTX_over_dim_XTX" : actual_nu_XTX_over_dim_XTX,
    "estimate_nu_XTX_over_dim_XTX" : estimate_nu_XTX_over_dim_XTX,
    "actual_alpha_XTX" : actual_alpha_XTX, 
    "estimate_alpha_XTX" : estimate_alpha_XTX,
    "actual_nu_XXT_over_dim_XXT" :  actual_nu_XXT_over_dim_XXT,
    "estimate_nu_XXT_over_dim_XXT" : estimate_nu_XXT_over_dim_XXT, 
    "actual_alpha_XXT" :  actual_alpha_XXT,
    "estimate_alpha_XXT" : estimate_alpha_XXT
    }
    
  return results_dict
    
def experiment_relu_X_gelu_X_tanh_X(N : int, d : int, alpha : float, verbose = False) -> tuple:
  
  X = generate_data_manifold(N, d, alpha)
  dim_X = X.shape[1]
  actual_nu_X_over_dim_X = calculate_PatnaikPearson_dim(X) / dim_X
  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(actual_nu_X_over_dim_X, dim_X)
  if verbose:
    print("alpha = ", alpha, ", actual_alpha_X = ", actual_alpha_X)
    print("actual_nu_X_over_dim_X = ", actual_nu_X_over_dim_X)

  reluX = this_vec_relu(X)
  k = 3
  if verbose: 
    print(X[0:k,0:k])
    print(reluX[0:k,0:k])
  dim_reluX = reluX.shape[1]
  actual_nu_reluX_over_dim_reluX = calculate_PatnaikPearson_dim(reluX) / dim_reluX
  actual_alpha_reluX = calculate_alpha_given_nu_over_d_and_d(actual_nu_reluX_over_dim_reluX, dim_reluX)
  if verbose:
    print("actual_alpha_reluX = ", actual_alpha_reluX)
    print("actual_nu_reluX_over_dim_reluX = ", actual_nu_reluX_over_dim_reluX)

  
  geluX = gelu_approx(X)
  k = 3
  if verbose: 
    print(X[0:k,0:k])
    print(geluX[0:k,0:k])
  dim_geluX = geluX.shape[1]
  actual_nu_geluX_over_dim_geluX = calculate_PatnaikPearson_dim(geluX) / dim_geluX
  actual_alpha_geluX = calculate_alpha_given_nu_over_d_and_d(actual_nu_geluX_over_dim_geluX, dim_geluX)
  if verbose:
    print("actual_alpha_geluX = ", actual_alpha_geluX)
    print("actual_nu_geluX_over_dim_geluX = ", actual_nu_geluX_over_dim_geluX)

  tanhX = this_vec_tanh(X)
  k = 3
  if verbose: 
    print(X[0:k,0:k])
    print(tanhX[0:k,0:k])
  dim_tanhX = tanhX.shape[1]
  actual_nu_tanhX_over_dim_tanhX = calculate_PatnaikPearson_dim(tanhX) / dim_tanhX
  actual_alpha_tanhX = calculate_alpha_given_nu_over_d_and_d(actual_nu_tanhX_over_dim_tanhX, dim_tanhX)
  if verbose:
    print("actual_alpha_tanhX = ", actual_alpha_tanhX)
    print("actual_nu_tanhX_over_dim_tanhX = ", actual_nu_tanhX_over_dim_tanhX)

  results_dict = { 
    "actual_nu_X_over_dim_X" :  actual_nu_X_over_dim_X,
    "actual_alpha_X" : actual_alpha_X, 
    "actual_nu_reluX_over_dim_reluX" : actual_nu_reluX_over_dim_reluX,
    "actual_alpha_reluX" :  actual_alpha_reluX,
    "actual_nu_geluX_over_dim_geluX" : actual_nu_geluX_over_dim_geluX,
    "actual_alpha_geluX" : actual_alpha_geluX, 
    "actual_nu_tanhX_over_dim_tanhX" : actual_nu_tanhX_over_dim_tanhX,
    "actual_alpha_tanhX" : actual_alpha_tanhX
    }
    
  return results_dict

# apply relu to an array by vectorisation
def this_relu(x):
  if x > 0.0:
    return x
  else:
    return 0.0

# Apply the function using np.vectorize
this_vec_relu = np.vectorize(this_relu)

# apply tanh to an array by vectorisation
def this_tanh(x):
  if x < 0:
    e_to_the_two_x = np.exp(2 * x)
    return (e_to_the_two_x - 1.0) / (e_to_the_two_x + 1.0)
  else:
    e_to_the_minus_two_x = np.exp(-2 * x)
    return (1.0 - e_to_the_minus_two_x) / (1.0 + e_to_the_minus_two_x)

# Apply the function using np.vectorize
this_vec_tanh = np.vectorize(this_tanh)

def this_sigmoid(x):
    return 0.5 * (this_tanh(0.5 * x) + 1)
    
# Apply the function using np.vectorize
this_vec_sigmoid = np.vectorize(this_sigmoid)

from scipy.special import erf

def gelu_exact(x: np.ndarray) -> np.ndarray:
    """
    Exact GELU activation using the error function.
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

def gelu_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximate GELU activation using tanh-based formula.
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
    ))
    
def calculate_nu_alpha_d(alpha : float,
                         d : int,
                         uniform_draws : bool = False
                         ) -> float:
                             
  # uses correct alpha
  # ** TO DO : adapt to GPU?

  these_sigmas = generate_pareto_draws(d, alpha, uniform_draws)

  return calculate_nu(these_sigmas)
  
def calculate_nu_alpha_d_old(alpha : float,
                         d : int,
                         uniform_draws : bool = False
                         ) -> float:
                             
  # uses correct alpha
  # ** TO DO : adapt to GPU?

  these_sigmas = np.zeros(d)
  for i in range(1,d+1): # 1 <= i <= d
    this_Fx = i / (d+1)
    if not uniform_draws: 
        this_Fx = np.random.uniform(0,1)
    this_x = (1.0/(1 - this_Fx))** (1.0/alpha)
    these_sigmas[i-1] = this_x

  return calculate_nu(these_sigmas)
  
def row_wise_softmax(input_array : np.ndarray,
                     temperature : float = 1.0
                     ) -> np.ndarray:

  # input_array is assumed to be N * d
  
  # ** TO DO : adapt to GPU?

  this_softmax = np.zeros(input_array.shape)
  num_rows = input_array.shape[0]
  for i in range(num_rows):
    row_i = input_array[i,:]
    this_softmax[i,:] = softmax(input_array[i,:], temperature)

  return this_softmax
  
def softmax(input_array : np.ndarray,
            temperature : float
            ) -> np.ndarray:
                
    # ** TO DO : adapt to GPU?

    """Compute softmax values for each sets of scores in x."""

    input_array = input_array / temperature
    e_input_array = np.exp(input_array - np.max(input_array)) # this fails if input_array is not 1-dimensional

    return e_input_array / e_input_array.sum()
    
def generate_square_weight_matrix(d : int,
                                  alpha : float,
                                  uniform_draws : bool = False,
                                  use_pareto : bool = True,
                                  use_uniform : bool = False,
                                  use_cauchy : bool = False,
                                  verbose : bool = False
                                  ) -> np.ndarray:

  # d is the dimension of the square matrix, which will be d * d
  # alpha is the tail exponent of the singular values of the matrix
  
  # uses correct alpha
  # ** TO DO : adapt to GPU?
  
  these_sigmas = np.zeros(d)
  if use_pareto:
    these_sigmas = generate_pareto_draws(d, alpha, uniform_draws)
    if verbose:
        print("using pareto")
        print(these_sigmas[0:5])
  elif use_uniform:
    if verbose:
        print("using uniform")
    these_sigmas = np.random.uniform(0,1,d)
  elif use_cauchy:
    if verbose:
        print("using cauchy")
    these_sigmas = np.random.standard_cauchy(d)
  else:
    print("*** error : need to set at least one of use_pareto, use_uniform, use_cauchy to True ***")
    
  if use_gpu:
    print(" ** generate_square_weight_matrix: using GPU **")
    Wdiag = cp.diag(these_sigmas)
    Wleft = cp.array(generate_orthogonal_matrix(d))
    Wright = cp.array(generate_orthogonal_matrix(d))
    W = cp.matmul(cp.matmul(Wleft, Wdiag), Wright)
    return cp.asnumpy(W)
  else:
    print(" ** generate_square_weight_matrix: using CPU **")
    Wdiag = np.diag(these_sigmas)
    Wleft = generate_orthogonal_matrix(d)
    Wright = generate_orthogonal_matrix(d)
    W = Wleft @ Wdiag @ Wright
    return W
  
def generate_data_manifold(N : int,
                           d : int,
                           alpha : float,
                           uniform_draws : bool = False,
                           use_pareto : bool = True,
                           use_uniform : bool = False,
                           use_cauchy : bool = False,
                           verbose : bool = False,
                           use_svd : bool = False
                           ) -> np.ndarray:

  # N is the number of points in our realisation X of our data manifold
  # d is the dimension of the ambient space
  # alpha is the tail exponent
  
  # uses correct alpha
  
  X0 = np.random.randn(N, d)

  these_sigmas = np.zeros(d)
  if use_pareto:
    these_sigmas = generate_pareto_draws(d, alpha, uniform_draws)
    if verbose:
        print("using pareto")
        print(these_sigmas[0:5])
  elif use_uniform:
    these_sigmas = np.random.uniform(0,1,d)
    if verbose:
        print("using uniform")
        print(these_sigmas[0:5])
  elif use_cauchy:
    these_sigmas = np.random.standard_cauchy(d)
    if verbose:
        print("using cauchy")
        print(these_sigmas[0:5])
  else:
    print("*** error : need to set at least one of use_pareto, use_uniform, use_cauchy to True ***")

  Xdiag = np.diag(these_sigmas)
  
  X = np.zeros((N,d))
  
  if use_svd:
    print("generate_data_manifold: using svd") 
    U = generate_orthogonal_matrix(N)
    Xdiag = np.zeros((N,d))
    for i in range(0, min(N,d)):
        Xdiag[i,i] = these_sigmas[i]
    VT = generate_orthogonal_matrix(d)
    if use_gpu:
        U_gpu = cp.array(U) # move to GPU
        Xdiag_gpu = cp.array(Xdiag)  # move to GPU 
        VT_gpu = cp.array(VT) # move to GPU
        X_gpu = cp.matmul(cp.matmul(U_gpu, Xdiag_gpu), VT_gpu)  # runs on GPU
        X = cp.asnumpy(X_gpu)  # move back to numpy
    else:
        X = U @ Xdiag @ VT      
  else: 
    if use_gpu:
        if verbose:
            print("running on GPU")
        X0_gpu = cp.array(X0)  # move to GPU
        Xdiag_gpu = cp.array(Xdiag)  # move to GPU 
        X_gpu = cp.matmul(X0_gpu, Xdiag_gpu)  # runs on GPU
        X = cp.asnumpy(X_gpu)  # move back to numpy
    else:
        if verbose:
            print("running on CPU")
        X = X0 @ Xdiag

  return X
  
def calculate_nu(these_lambdas : np.array, 
                 force_cpu : bool = False
                 ) -> float:
    
  #use_gpu = torch.cuda.is_available():
  if use_gpu and not force_cpu:
      return calculate_nu_gpu(cp.array(these_lambdas))
  else:
      return calculate_nu_cpu(these_lambdas)
      
  
def calculate_nu_cpu(these_lambdas : np.array) -> float:
  print("** calculate_nu_cpu ** running on CPU")
  sum_lambda_i = sum(these_lambdas)
  sum_lambda_i_squared = sum(these_lambdas**2)
  nu = (sum_lambda_i ** 2) / sum_lambda_i_squared
  return nu
  
def calculate_nu_gpu(these_lambdas : cp.array) -> float:
  #print("** calculate_nu_gpu ** running on GPU")
  sum_lambda_i = cp.sum(these_lambdas)
  sum_lambda_i_squared = cp.sum(these_lambdas**2)
  nu = (sum_lambda_i ** 2) / sum_lambda_i_squared
  return nu
  
def calculate_nu_s(s : float, 
                   these_lambdas : np.array
                   ) -> float:
  # generalised nu                      
  lambdas_to_the_s = this_vec_raise_to_power(these_lambdas,s)
  return calculate_nu(lambdas_to_the_s)


# raise to the power s, for some s > 0
def raise_to_power(x : float, 
                   s : float
                   ) -> float:                     
  # raise x to the power s
  return x**s

# vectorize
this_vec_raise_to_power = np.vectorize(raise_to_power)

def inv_pareto_cdf(alpha : float, 
            y : float
            ) -> float:
                
    # now uses correct value of alpha 
    
    eps = 1e-6
    #this_exponent = 1.0 /(alpha - 1.0) # OLD
    this_exponent = 1.0 /alpha # NEW
    one_over_one_minus_y = 1.0 / max(eps,1.0 - max(eps,y))
    return one_over_one_minus_y ** this_exponent

# Apply the function using np.vectorize
this_vec_inv_pareto_cdf = np.vectorize(inv_pareto_cdf)
  
def calculate_stable_rank(these_lambdas : np.array) -> float:
    
  # adapted to GPU
  
  frobenius_norm_squared = 0.0
  operator_norm_squared = 0.0
  
  if use_gpu:
    cp_these_lambdas = cp.array(these_lambdas)
    these_lambdas_squared = cp_these_lambdas ** 2
    frobenius_norm_squared = cp.sum(these_lambdas_squared)
    operator_norm_squared = cp.max(these_lambdas_squared)
  else: # CPU  
    these_lambdas_squared = these_lambdas ** 2
    frobenius_norm_squared = sum(these_lambdas_squared)
    operator_norm_squared = max(these_lambdas_squared)
    
  eps = 1e-6
  stable_rank = 0.0
  if operator_norm_squared > eps:
    stable_rank = frobenius_norm_squared / operator_norm_squared
  return stable_rank
      
  
def generate_orthogonal_matrix(dim : int) -> np.ndarray:
  """
  generate an orthogonal matrix of dimension dim, using the QR decomposition. 
  """
  
  H = np.random.randn(dim, dim)
  Q, R = np.linalg.qr(H)
  Q *= np.sign(np.diag(R)) # Ensure positive diagonal
  return Q
  
def calculate_nu_twonn_dim(
	input_data : np.ndarray,
	verbose : bool = False
	) -> tuple[float, float]:
        
  # ** TO DO : adapt to GPU?


  X = input_data
  this_N = X.shape[0]
  this_d = X.shape[1]
  if verbose:
    plot_histogram_of_values(X, verbose=True, rescale_factor=1.0, num_bins=100)

  # calculate mu, the average of the N column vectors
  mu = X.sum(axis=0) / this_N
  if verbose:
    plot_histogram_of_values(mu, verbose=True, rescale_factor=1.0, num_bins=100)

  Xdemeaned = X - mu
  if verbose:
    plot_histogram_of_values(Xdemeaned, verbose=True, rescale_factor=1.0, num_bins=100)

  U, S, Vh = np.linalg.svd(Xdemeaned, full_matrices=True)

  # Print the shapes of the resulting matrices
  if verbose:
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)

  # Reconstruct the original matrix
  Sigma = np.zeros((this_N, this_d))
  for i in range(this_d):
    Sigma[i, i] = S[i]
  Xdemeaned_reconstructed = U @ Sigma @ Vh

  # Verify the reconstruction
  if verbose:
    print("Reconstruction error:", np.linalg.norm(Xdemeaned - Xdemeaned_reconstructed))

  if verbose:
    plot_histogram_of_values(S)

  nu = calculate_nu(S)
  if verbose:
    print("nu = ", nu)

  twonn_dim = twonn_intrinsic_dimension(X, plot_fit=verbose)

  return nu, twonn_dim
  
def calculate_PatnaikPearson_dim(
	input_data : np.ndarray,
	verbose : bool = False
	) -> float:
        
  if use_gpu:
      return calculate_PatnaikPearson_dim_gpu(input_data, verbose)
  else:
      return calculate_PatnaikPearson_dim_cpu(input_data, verbose)
      
  
def calculate_PatnaikPearson_dim_cpu(
	input_data : np.ndarray,
	verbose : bool = False
	) -> float:
        
  if verbose:
    print("reached calculate_PatnaikPearson_dim_cpu")

  X = input_data
  this_N = X.shape[0]
  this_d = X.shape[1]
  if verbose:
    plot_histogram_of_values(X, verbose=True, rescale_factor=1.0, num_bins=100)

  # calculate mu, the average of the N column vectors
  mu = X.sum(axis=0) / this_N
  if verbose:
    plot_histogram_of_values(mu, verbose=True, rescale_factor=1.0, num_bins=100)

  Xdemeaned = X - mu
  if verbose:
    plot_histogram_of_values(Xdemeaned, verbose=True, rescale_factor=1.0, num_bins=100)

  U, S, Vh = np.linalg.svd(Xdemeaned, full_matrices=True)

  # Print the shapes of the resulting matrices
  if verbose:
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)

  # Reconstruct the original matrix
  Sigma = np.zeros((this_N, this_d))
  for i in range(min(this_N,this_d)):
    Sigma[i, i] = S[i]
  Xdemeaned_reconstructed = U @ Sigma @ Vh

  # Verify the reconstruction
  if verbose:
    print("Reconstruction error:", np.linalg.norm(Xdemeaned - Xdemeaned_reconstructed))

  if verbose:
    plot_histogram_of_values(S)

  patnaik_pearson_dim = calculate_nu(S)
  if verbose:
    print("patnaik_pearson_dim = ", patnaik_pearson_dim)

  return patnaik_pearson_dim
  
  
def calculate_PatnaikPearson_dim_gpu(
	input_data : np.ndarray,
	verbose : bool = False
	) -> float:
        
  if verbose:
      print("reached calculate_PatnaikPearson_dim_gpu")
      print("input_data.shape = ",input_data.shape)
      display_stats(input_data)

  X = cp.array(input_data)
  this_N = X.shape[0]
  this_d = X.shape[1]
  if verbose:
    plot_histogram_of_values(input_data, verbose=True, rescale_factor=1.0, num_bins=100)

  # calculate mu, the average of the N column vectors
  mu = X.sum(axis=0) / this_N
  if verbose:
    plot_histogram_of_values(cp.asnumpy(mu), verbose=True, rescale_factor=1.0, num_bins=100)

  Xdemeaned = X - mu
  if verbose:
    plot_histogram_of_values(cp.asnumpy(Xdemeaned), verbose=True, rescale_factor=1.0, num_bins=100)

  U, S, Vh = cp.linalg.svd(Xdemeaned, full_matrices=True)

  # Print the shapes of the resulting matrices
  if verbose:
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)

  # Reconstruct the original matrix
  Sigma = cp.zeros((this_N, this_d))
  for i in range(min(this_N,this_d)):
    Sigma[i, i] = S[i]
  Xdemeaned_reconstructed = cp.matmul(cp.matmul(U, Sigma), Vh)

  # Verify the reconstruction
  if verbose:
    print("Reconstruction error:", cp.linalg.norm(Xdemeaned - Xdemeaned_reconstructed))

  if verbose:
    plot_histogram_of_values(cp.asnumpy(cp.linalg.norm(S)))

  patnaik_pearson_dim = calculate_nu_gpu(S)
  if verbose:
    print("patnaik_pearson_dim = ", patnaik_pearson_dim)

  return patnaik_pearson_dim
  
  
def calculate_PatnaikPearson_dim_for_MP(d : int, 
                                        aspect_ratio : float,
                                        force_cpu : bool = False
                                        ) -> float:

  N = (int) (d / aspect_ratio)
  
  if use_gpu and not force_cpu:
    X = cp.random.normal(loc=0.0, scale=1.0, size=(d, N))
    Yd = cp.asnumpy((1.0 / N) * cp.matmul( X, X.T))
    return calculate_PatnaikPearson_dim(Yd)
  else:
    X = np.random.normal(loc=0.0, scale=1.0, size=(d, N))
    Yd = (1.0 / N) * X @ X.T
    return calculate_PatnaikPearson_dim(Yd)
  
def twonn_intrinsic_dimension(X : np.ndarray,
                               plot_fit : bool = True,
                               use_cosine_similarity : bool = False,
                               discard_proportion : float = 0.0
                               ) -> float:
                                   
    # ** TO DO : adapt to GPU?
                                   
    """
    Estimates the Intrinsic Dimension (ID) of a dataset using the TWO-NN algorithm.
    Args:
        X (numpy.ndarray): Input data array of shape (n_samples, n_features).
        plot_fit (bool): If True, plots the linear regression fit.
    Returns:
        float: The estimated intrinsic dimension.


    originally from https://data-processing.club/twonn/
    modified by me to allow for use of cosine similarity as well as euclidean distance
    """

    N = X.shape[0]
    r1 = np.zeros((N))
    r2 = np.zeros((N))

    # ---------------------------------------------------------
    # Step 1: Compute pairwise distances and find
    # the first two nearest neighbors.
    # ---------------------------------------------------------
    if use_cosine_similarity:
      XTX = np.matmul(X, X.T)
      diag_inv_norms = np.zeros((N,N))
      eps = 1e-6
      for i in range(0,N):
          diag_inv_norms[i,i] = 1/ np.sqrt(eps + XTX[i,i])
      cosine_similarities = diag_inv_norms @ XTX @ diag_inv_norms
      for i in range(0,N):
          similarities_i = np.sort(cosine_similarities[i,:])[::-1]
          r1[i] = similarities_i[1]
          r2[i] = similarities_i[2]
      r1 = np.arccos(r1)
      r2 = np.arccos(r2)
    else: # use kNN
      # We need k=3 because the 1st neighbor is the point itself (distance 0).
      # So, index 1 is the 1st NN, index 2 is the 2nd NN.
      nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
      distances, indices = nbrs.kneighbors(X)
      # Extract r1 (1st neighbor dist) and r2 (2nd neighbor dist)
      # distances[:, 0] is the distance to self (0.0)
      r1 = distances[:, 1]
      r2 = distances[:, 2]

    mu = r2 / r1

    # ---------------------------------------------------------
    # Step 2: Compute the empirical Cumulate Distribution Function (CDF).
    # ---------------------------------------------------------
    # Sort mu values in ascending order.
    mu_sorted = np.sort(mu)

    # The empirical CDF F(mu_i) is defined as i / N.
    F_emp = np.arange(N) / N

    if discard_proportion > 0.0:
      desired_length = int(N * (1.0 - discard_proportion))
      mu_sorted = mu_sorted[:desired_length]
      F_emp = F_emp[:desired_length]

    # ---------------------------------------------------------
    # Step 3: Linear Regression to find the dimension d.
    # The theory states: -log(1 - F(mu)) = d * log(mu)
    # ---------------------------------------------------------
    # X axis: log(mu)
    x = np.log(mu_sorted)

    # Y axis: -log(1 - F(mu))
    y = -np.log(1 - F_emp)

    # Perform Linear Regression passing through the origin.
    # Model: y = slope * x
    # The closed-form solution for slope (d) minimizing sum of squared errors
    # for a line through origin is: sum(x*y) / sum(x^2).
    slope = np.dot(x, y) / np.dot(x, x)
    intrinsic_dim = slope

    # ---------------------------------------------------------
    # Optional: Visualization of the fit
    # ---------------------------------------------------------
    if plot_fit:
        plt.figure(figsize=(8, 6))

        # Plot all points (grey for discarded ones)
        plt.scatter(x, y, s=5, color='#005aff', label='Data points')

        # Plot the fitted line
        # Create a line from origin to max x used
        line_x = np.linspace(0, np.max(x), 100)
        line_y = slope * line_x
        plt.plot(line_x, line_y, color='#ff4b00', linewidth=2, label=f'Fit (Slope = ID = {slope:.2f})')

        plt.xlabel('ln(mu)', fontsize=20)
        plt.ylabel('-ln(1 - F(mu))', fontsize=20)
        plt.title(f'TWO-NN Intrinsic Dimension Estimation\nEstimated ID: {intrinsic_dim:.4f}', fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return intrinsic_dim
   
   

def plot_histogram_of_eigenvalues(input_data : np.ndarray,
                                  verbose : bool = False,
                                  rescale_factor : float = 1.0,
                                  num_bins : int = 100):

  eig_vals, eig_vecs = np.linalg.eig(input_data)

  print("summary stats for eigenvalues")
  print("type :", type(eig_vals))
  print("shape :", eig_vals.shape)
  print("min :", np.min(eig_vals))
  print("max :", np.max(eig_vals))
  print("mean :", np.mean(eig_vals))
  print("std :", np.std(eig_vals))

  custom_bins = []
  this_min = np.min(eig_vals)
  this_max = np.max(eig_vals)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  flattened_array = eig_vals.flatten()
  print("flattened_array.shape = ",flattened_array.shape)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(flattened_array, bins=custom_bins)

  print("Histogram counts:", hist)

  # Plot histogram
  plt.hist(flattened_array, bins=custom_bins, edgecolor='black')
  plt.title("Eigenvalues")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()

  # Plot histogram
  plt.hist(flattened_array, bins=custom_bins, edgecolor='black')
  plt.xscale('log')
  plt.yscale('log')
  plt.title("Log/log plot : Eigenvalues")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()

  return
  
def clean_array(this_array : list, num_decimal_places : int = 3) -> list:
  # TO DO : why are all the names here "array" when this deals with lists?
  new_array = []
  scale_factor = 10**num_decimal_places
  for val in this_array:
    new_val = ((int) (val * scale_factor)) / scale_factor
    new_array.append(new_val)
  return new_array
  
def plot_histogram_of_values( input_data : np.ndarray,
                              verbose : bool = False,
                              rescale_factor : float = 1.0,
                              num_bins : int = 100,
                              title_text : str = "",
                              image_name : str = "test"):

  input_values = input_data.flatten()

  print("summary stats for input values")
  print("type :", type(input_values))
  print("shape :", input_values.shape)
  print("min :", np.min(input_values))
  print("max :", np.max(input_values))
  num_dp = 4
  this_mean = (clean_array([np.mean(input_values)],num_dp))[0]
  this_std = (clean_array([np.std(input_values)],num_dp))[0]
  print("mean :", this_mean)
  print("std :", this_std)

  custom_bins = []
  this_min = np.min(input_values)
  this_max = np.max(input_values)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  #num_bins = 100
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(input_values, bins=custom_bins)

  print("Histogram counts:", hist)

  # Plot histogram
  this_title = title_text + ", mean = " + str(this_mean) + ", std = " + str(this_std)
  plt.hist(input_values, bins=custom_bins, edgecolor='black')
  plt.title(this_title)
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.savefig(image_name + '.pdf', dpi=300, bbox_inches='tight')
  plt.show()

  # Plot histogram
  plt.hist(input_values, bins=custom_bins, edgecolor='black')
  plt.xscale('log')
  plt.yscale('log')
  plt.title("Log/log plot : input values")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()

  return
  
def plot_histogram_of_values_older( input_data : np.ndarray,
                              verbose : bool = False,
                              rescale_factor : float = 1.0,
                              num_bins : int = 100):

  input_values = input_data.flatten()

  print("summary stats for input values")
  print("type :", type(input_values))
  print("shape :", input_values.shape)
  print("min :", np.min(input_values))
  print("max :", np.max(input_values))
  print("mean :", np.mean(input_values))
  print("std :", np.std(input_values))

  custom_bins = []
  this_min = np.min(input_values)
  this_max = np.max(input_values)
  this_mid = 0.5 * (this_min + this_max)
  min_value = this_mid + 1.25 * rescale_factor * (this_min - this_mid)
  max_value = this_mid + 1.25 * rescale_factor * (this_max - this_mid)
  step_size = (max_value - min_value) / num_bins
  for i in range(num_bins + 1):
    custom_bins.append(min_value + i * step_size)

  # Compute histogram with custom bins
  hist, bin_edges = np.histogram(input_values, bins=custom_bins)

  print("Histogram counts:", hist)

  # Plot histogram
  plt.hist(input_values, bins=custom_bins, edgecolor='black')
  plt.title("input values")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()

  # Plot histogram
  plt.hist(input_values, bins=custom_bins, edgecolor='black')
  plt.xscale('log')
  plt.yscale('log')
  plt.title("Log/log plot : input values")
  plt.xlabel("Bins")
  plt.ylabel("Frequency")
  plt.show()

  return
  
def estimate_product_nu_over_d(nu_over_d_1 : float,
                               d_1 : int,
                               nu_over_d_2 : float,
                               d_2 : int
                               ) -> float:
                                   
  # ** TO DO : use new alpha - check the interpolation thresholds                               
                               
  # calculate alpha_1 corresponding to nu_over_d_1, and alpha_2 corresponding to nu_over_d_2
  # calculate the corresponding product_alpha
  # calculate the corresponding nu_over_d
  
  print("** estimate_product_nu_over_d: use new alpha - check the interpolation thresholds **")

  #return estimate_product_nu_over_d_base(nu_over_d_1, d_1, nu_over_d_2, d_2, lower_interpolation_threshold=1.5, upper_interpolation_threshold=2.6)
  
  return estimate_product_nu_over_d_base(nu_over_d_1, d_1, nu_over_d_2, d_2, lower_interpolation_threshold=0.5, upper_interpolation_threshold=1.6)

def estimate_product_nu_over_d_base(nu_over_d_1 : float,
                                    d_1 : int,
                                    nu_over_d_2 : float,
                                    d_2 : int,
                                    lower_interpolation_threshold : float,
                                    upper_interpolation_threshold : float
                                    ) -> float:
  # calculate alpha_1 corresponding to nu_over_d_1, and alpha_2 corresponding to nu_over_d_2
  # calculate the corresponding product_alpha
  # calculate the corresponding nu_over_d

  alpha_1 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_1, d_1)
  alpha_2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_2, d_2)
  product_alpha = estimate_product_alpha_base(alpha_1, d_1, alpha_2, d_2, lower_interpolation_threshold, upper_interpolation_threshold)
  if d_1 != d_2:
      print("** warning : estimate_product_nu_over_d_base implicitly assumes that d_1 = d_2, whereas d_1 = " + str(d_1) + ", d_2 = " + str(d_2))
      return 0.0
  product_nu_over_d = calculate_nu_over_d_given_alpha_and_d(product_alpha, d_1)

  return product_nu_over_d

def estimate_product_alpha(alpha_1 : float,
                           d_1 : int,
                           alpha_2 : float,
                           d_2 : int
                           ) -> float:

  #return estimate_product_alpha_base(alpha_1, d_1, alpha_2, d_2, lower_interpolation_threshold=1.5, upper_interpolation_threshold=2.6)
  return estimate_product_alpha_base(alpha_1, d_1, alpha_2, d_2, lower_interpolation_threshold=0.5, upper_interpolation_threshold=1.6)

def estimate_product_alpha_base(alpha_1 : float,
                                d_1 : int,
                                alpha_2 : float,
                                d_2 : int,
                                lower_interpolation_threshold : float = 0.5,
                                #lower_interpolation_threshold : float = 1.5,
                                upper_interpolation_threshold : float = 1.6
                                #upper_interpolation_threshold : float = 2.6
                           ) -> float:
                               
                              
  # ** TO DO : use new alpha
  # ** TO DO : adapt to GPU?

  # if min(alpha_1, alpha_2) < lower, then product_alpha is min(alpha_1, alpha_2)
  # if min(alpha_1, alpha_2) > upper, then, calculate the corresponding nu_over_d 's, take the product of these, find the corresponding alpha
  # for lower < min(alpha_1, alpha_2) < upper, then it's a limear interpolation between the two
  # we also expect to have d_1 == d_2, since this is for multiplying two square matrices together

  min_alpha = min(alpha_1, alpha_2)
  if min_alpha < lower_interpolation_threshold:
    return min_alpha

  nu_over_d_1 = calculate_nu_over_d_given_alpha_and_d(alpha_1, d_1)
  nu_over_d_2 = calculate_nu_over_d_given_alpha_and_d(alpha_2, d_2)
  product_nu_over_d = nu_over_d_1 * nu_over_d_2
  product_alpha = calculate_alpha_given_nu_over_d_and_d(product_nu_over_d, min(d_1,d_2))

  if min_alpha > upper_interpolation_threshold:
    return product_alpha

  interpolation_parameter = (min_alpha - lower_interpolation_threshold) / (upper_interpolation_threshold - lower_interpolation_threshold)
  interpolation_parameter = min(1.0, max(0.0, interpolation_parameter))
  result = (1.0 - interpolation_parameter) * min_alpha + interpolation_parameter * product_alpha

  return result
  
def calculate_nu_over_d_given_alpha_and_d(alpha : float, d : int) -> float:
    
    # ** TO DO : use new alpha
    # ** TO DO : adapt to GPU?
    
    _, nu_over_d = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha, d)
    return nu_over_d
  
def calculate_nu_over_d_given_alpha(alpha : float) -> float:
    
  print("** need to deprecate calculate_nu_over_d_given_alpha : replace with calculate_nu_over_d_given_alpha_and_d !! **")

  alpha_vals, nu_over_d_vals = get_alpha_vals_nu_over_d_vals()

  if alpha <= alpha_vals[0]:
    return nu_over_d_vals[0]

  for i in range(len(alpha_vals)):
    if alpha_vals[i] >= alpha:
      return nu_over_d_vals[i]

  return nu_over_d_vals[-1]
  
def calculate_alpha_given_nu_over_d(nu_over_d : float) -> float:
    
  print("** deprecate use of calculate_alpha_given_nu_over_d : use calculate_alpha_given_nu_over_d_and_d instead! **")

  alpha_vals, nu_over_d_vals = get_alpha_vals_nu_over_d_vals()

  if nu_over_d <= nu_over_d_vals[0]:
    return alpha_vals[0]

  for i in range(len(nu_over_d_vals)):
    if nu_over_d_vals[i] >= nu_over_d:
      return alpha_vals[i]

  return alpha_vals[-1]
  
def get_alpha_vals_nu_over_d_vals() -> tuple[np.ndarray, np.ndarray]:
    
  print("** deprecate use of get_alpha_vals_nu_over_d_vals !! **")

  smoothed_alpha_vals =  [1.109719, 1.113363, 1.117818, 1.123082, 1.129075, 1.135717, 1.142925, 1.150619, 1.158718, 1.167141, 1.175807, 1.184635, 1.193544, 1.202452, 1.211361, 1.22027, 1.229179, 1.238088, 1.246997, 1.255906, 1.264815, 1.273724, 1.282633, 1.291542, 1.30045 , 1.309359, 1.318268, 1.327177, 1.336086, 1.344995, 1.353904, 1.362813, 1.371722, 1.380631, 1.38954 , 1.398448, 1.407357, 1.416266, 1.425175, 1.434084, 1.442993, 1.451902, 1.460811, 1.46972 , 1.478629, 1.487538, 1.496446, 1.505355, 1.514264, 1.523173, 1.532082, 1.540991, 1.5499  , 1.558809, 1.567718, 1.576627, 1.585536, 1.594444, 1.603353, 1.612262, 1.621171, 1.63008 , 1.638989, 1.647898, 1.656807, 1.665716, 1.674625, 1.683534, 1.692442, 1.701351, 1.71026 , 1.719169, 1.728078, 1.736987, 1.745896, 1.754805, 1.763714, 1.772623, 1.781532, 1.79044, 1.799349, 1.808258, 1.817167, 1.826076, 1.834985, 1.843894, 1.852803, 1.861712, 1.870621, 1.87953 , 1.888438, 1.897347, 1.906256, 1.915165, 1.924074, 1.932983, 1.941892, 1.950801, 1.95971 , 1.968619, 1.977528, 1.986436, 1.995345, 2.004254, 2.013163, 2.022072, 2.030981, 2.03989 , 2.048799, 2.057708, 2.066617, 2.075526, 2.084434, 2.093343, 2.102252, 2.111161, 2.12007 , 2.128979, 2.137888, 2.146797, 2.155706, 2.164615, 2.173524, 2.182432, 2.191341, 2.20025 , 2.209159, 2.218068, 2.226977, 2.235886, 2.244795, 2.253704, 2.262613, 2.271522, 2.28043 , 2.289339, 2.298248, 2.307157, 2.316066, 2.324975, 2.333884, 2.342793, 2.351702, 2.360611, 2.36952 , 2.378428, 2.387337, 2.396246, 2.405155, 2.414064, 2.422973, 2.431882, 2.440791, 2.4497  , 2.458609, 2.467518, 2.476426, 2.485335, 2.494244, 2.503153, 2.512062, 2.520971, 2.52988 , 2.538789, 2.547698, 2.556607, 2.565516, 2.574424, 2.583333, 2.592242, 2.601151, 2.61006 , 2.618969, 2.627878, 2.636787, 2.645696, 2.654605, 2.663514, 2.672422, 2.681331, 2.69024 , 2.699149, 2.708058, 2.716967, 2.725876, 2.734785, 2.743694, 2.752603, 2.761512, 2.77042 , 2.779329, 2.788238, 2.797147, 2.806056, 2.814965, 2.823874, 2.832783, 2.841692, 2.850601, 2.85951, 2.868418, 2.877327, 2.886236, 2.895145, 2.904054, 2.912963, 2.921872, 2.930781, 2.93969 , 2.948599, 2.957508, 2.966416, 2.975325, 2.984234, 2.993143, 3.002052, 3.010961, 3.01987 , 3.028779, 3.037688, 3.046597, 3.055506, 3.064414, 3.073323, 3.082232, 3.091141, 3.10005 , 3.108959, 3.117868, 3.126777, 3.135686, 3.144595, 3.153504, 3.162412, 3.171321, 3.18023 , 3.189139, 3.198048, 3.206957, 3.215866, 3.224775, 3.233684, 3.242593, 3.251502, 3.26041 , 3.269319, 3.278228, 3.287137, 3.296046, 3.304955, 3.313864, 3.322773, 3.331682, 3.340591, 3.349499, 3.358408, 3.367317, 3.376226, 3.385135, 3.394044, 3.402953, 3.411862, 3.420771, 3.42968, 3.438589, 3.447497, 3.456406, 3.465315, 3.474224, 3.483133, 3.492042, 3.500951, 3.50986 , 3.518769, 3.527678, 3.536587, 3.545495, 3.554404, 3.563313, 3.572222, 3.581131, 3.59004 , 3.598949, 3.607858, 3.616767, 3.625676, 3.634585, 3.643493, 3.652402, 3.661311, 3.67022 , 3.679129, 3.688038, 3.696947, 3.705856, 3.714765, 3.723674, 3.732583, 3.741491, 3.7504  , 3.759309, 3.768218, 3.777127, 3.786036, 3.794945, 3.803854, 3.812763, 3.821672, 3.830581, 3.839489, 3.848398, 3.857307, 3.866216, 3.875125, 3.884034, 3.892943, 3.901852, 3.910761, 3.91967 , 3.928579, 3.937487, 3.946396, 3.955305, 3.964214, 3.973123, 3.982032, 3.990941, 3.99985, 4.008759, 4.017668, 4.026577, 4.035485, 4.044394, 4.053303, 4.062212, 4.071121, 4.08003 , 4.088939, 4.097848, 4.106757, 4.115666, 4.124575, 4.133483, 4.142392, 4.151301, 4.16021 , 4.169119, 4.178028, 4.186937, 4.195846, 4.204755, 4.213664, 4.222573, 4.231481, 4.24039 , 4.249299, 4.258208, 4.267117, 4.276026, 4.284935, 4.293844, 4.302753, 4.311662, 4.320571, 4.329479, 4.338388, 4.347297, 4.356206, 4.365115, 4.374024, 4.382933, 4.391842, 4.400751, 4.40966 , 4.418569, 4.427477, 4.436386, 4.445295, 4.454204, 4.463113, 4.472022, 4.480931, 4.48984 , 4.498749, 4.507658, 4.516567, 4.525475, 4.534384, 4.543293, 4.552202, 4.561111, 4.57002, 4.578929, 4.587838, 4.596747, 4.605656, 4.614565, 4.623473, 4.632382, 4.641291, 4.6502  , 4.659109, 4.668018, 4.676927, 4.685836, 4.694745, 4.703654, 4.712563, 4.721471, 4.73038 , 4.739289, 4.748198, 4.757107, 4.766016, 4.774925, 4.783834, 4.792743, 4.801652, 4.810561, 4.819469, 4.828378, 4.837287, 4.846196, 4.855105, 4.864014, 4.872923, 4.881832, 4.890741, 4.89965 , 4.908559, 4.917467, 4.926376, 4.935285, 4.944194, 4.953103, 4.962012, 4.970921, 4.97983 , 4.988739, 4.997648, 5.006557, 5.015465, 5.024374, 5.033283, 5.042192, 5.051101, 5.06001 , 5.068919, 5.077828, 5.086737, 5.095646, 5.104555, 5.113463, 5.122372, 5.131281, 5.14019, 5.149099, 5.158008, 5.166917, 5.175826, 5.184735, 5.193644, 5.202553, 5.211461, 5.22037 , 5.229279, 5.238188, 5.247097, 5.256006, 5.264915, 5.273824, 5.282733, 5.291642, 5.300551, 5.309459, 5.318368, 5.327277, 5.336186, 5.345095, 5.354004, 5.362913, 5.371822, 5.380731, 5.38964 , 5.398549, 5.407457, 5.416366, 5.425275, 5.434184, 5.443093, 5.452002, 5.460911, 5.46982 , 5.478729, 5.487638, 5.496547, 5.505455, 5.514364, 5.523273, 5.532182, 5.541091, 5.55    , 5.558909, 5.567818, 5.576727, 5.585636, 5.594545, 5.603453, 5.612362, 5.621271, 5.63018 , 5.639089, 5.647998, 5.656907, 5.665816, 5.674725, 5.683634, 5.692543, 5.701451, 5.71036, 5.719269, 5.728178, 5.737087, 5.745996, 5.754905, 5.763814, 5.772723, 5.781632, 5.790541, 5.799449, 5.808358, 5.817267, 5.826176, 5.835085, 5.843994, 5.852903, 5.861812, 5.870721, 5.87963 , 5.888539, 5.897447, 5.906356, 5.915265, 5.924174, 5.933083, 5.941992, 5.950901, 5.95981 , 5.968719, 5.977628, 5.986537, 5.995445, 6.004354, 6.013263, 6.022172, 6.031081, 6.03999 , 6.048899, 6.057808, 6.066717, 6.075626, 6.084535, 6.093443, 6.102352, 6.111261, 6.12017 , 6.129079, 6.137988, 6.146897, 6.155806, 6.164715, 6.173624, 6.182533, 6.191441, 6.20035 , 6.209259, 6.218168, 6.227077, 6.235986, 6.244895, 6.253804, 6.262713, 6.271622, 6.280531, 6.289439, 6.298348, 6.307257, 6.316166, 6.325075, 6.333984, 6.342893, 6.351802, 6.360711, 6.36962 , 6.378529, 6.387437, 6.396346, 6.405255, 6.414164, 6.423073, 6.431982, 6.440891, 6.4498  , 6.458709, 6.467618, 6.476527, 6.485435, 6.494344, 6.503253, 6.512162, 6.521071, 6.52998 , 6.538889, 6.547798, 6.556707, 6.565616, 6.574525, 6.583433, 6.592342, 6.601251, 6.61016 , 6.619069, 6.627978, 6.636887, 6.645796, 6.654705, 6.663614, 6.672523, 6.681431, 6.69034 , 6.699249, 6.708158, 6.717067, 6.725976, 6.734885, 6.743794, 6.752703, 6.761612, 6.770521, 6.779429, 6.788338, 6.797247, 6.806156, 6.815065, 6.823974, 6.832883, 6.841792, 6.850701, 6.85961 , 6.868519, 6.877427, 6.886336, 6.895245, 6.904154, 6.913063, 6.921972, 6.930881, 6.93979 , 6.948699, 6.957608, 6.966517, 6.975425, 6.984334, 6.993243, 7.002152, 7.011061, 7.01997 , 7.028879, 7.037788, 7.046697, 7.055606, 7.064515, 7.073423, 7.082332, 7.091241, 7.10015 , 7.109059, 7.117968, 7.126877, 7.135786, 7.144695, 7.153604, 7.162513, 7.171421, 7.18033 , 7.189239, 7.198148, 7.207057, 7.215966, 7.224875, 7.233784, 7.242693, 7.251602, 7.260511, 7.269419, 7.278328, 7.287237, 7.296146, 7.305055, 7.313964, 7.322873, 7.331782, 7.340691, 7.3496, 7.358509, 7.367417, 7.376326, 7.385235, 7.394144, 7.403053, 7.411962, 7.420871, 7.42978 , 7.438689, 7.447598, 7.456507, 7.465415, 7.474324, 7.483233, 7.492142, 7.501051, 7.50996 , 7.518869, 7.527778, 7.536687, 7.545596, 7.554505, 7.563413, 7.572322, 7.581231, 7.59014 , 7.599049, 7.607958, 7.616867, 7.625776, 7.634685, 7.643594, 7.652503, 7.661411, 7.67032 , 7.679229, 7.688138, 7.697047, 7.705956, 7.714865, 7.723774, 7.732683, 7.741592, 7.750501, 7.759409, 7.768318, 7.777227, 7.786136, 7.795045, 7.803954, 7.812863, 7.821772, 7.830681, 7.83959 , 7.848498, 7.857407, 7.866316, 7.875225, 7.884134, 7.893043, 7.901952, 7.910861, 7.91977, 7.928679, 7.937588, 7.946496, 7.955405, 7.964314, 7.973223, 7.982132, 7.991041, 7.99995 , 8.008859, 8.017768, 8.026677, 8.035586, 8.044494, 8.053403, 8.062312, 8.071221, 8.08013 , 8.089039, 8.097948, 8.106857, 8.115766, 8.124675, 8.133584, 8.142492, 8.151401, 8.16031 , 8.169219, 8.178128, 8.187037, 8.195946, 8.204855, 8.213764, 8.222673, 8.231582, 8.24049 , 8.249399, 8.258308, 8.267217, 8.276126, 8.285035, 8.293944, 8.302853, 8.311762, 8.320671, 8.32958 , 8.338488, 8.347397, 8.356306, 8.365215, 8.374124, 8.383033, 8.391942, 8.400851, 8.40976 , 8.418669, 8.427578, 8.436486, 8.445395, 8.454304, 8.463213, 8.472122, 8.481031, 8.48994, 8.498849, 8.507758, 8.516667, 8.525576, 8.534484, 8.543393, 8.552302, 8.561211, 8.57012 , 8.579029, 8.587938, 8.596847, 8.605756, 8.614665, 8.623574, 8.632482, 8.641391, 8.6503  , 8.659209, 8.668118, 8.677027, 8.685936, 8.694845, 8.703754, 8.712663, 8.721572, 8.73048 , 8.739389, 8.748298, 8.757207, 8.766116, 8.775025, 8.783934, 8.792843, 8.801752, 8.810661, 8.81957 , 8.828478, 8.837387, 8.846296, 8.855205, 8.864114, 8.873023, 8.881932, 8.890841, 8.89975 , 8.908659, 8.917568, 8.926476, 8.935385, 8.944294, 8.953203, 8.962112, 8.971021, 8.97993 , 8.988839, 8.997748, 9.006657, 9.015566, 9.024474, 9.033383, 9.042292, 9.051201, 9.06011, 9.069019, 9.077928, 9.086837, 9.095746, 9.104655, 9.113564, 9.122472, 9.131381, 9.14029 , 9.149199, 9.158108, 9.167017, 9.175926, 9.184835, 9.193744, 9.202653, 9.211562, 9.22047 , 9.229379, 9.238288, 9.247197, 9.256106, 9.265015, 9.273924, 9.282833, 9.291742, 9.300651, 9.30956 , 9.318468, 9.327377, 9.336286, 9.345195, 9.354104, 9.363013, 9.371922, 9.380831, 9.38974 , 9.398649, 9.407558, 9.416466, 9.425375, 9.434284, 9.443193, 9.452102, 9.461011, 9.46992 , 9.478829, 9.487738, 9.496647, 9.505556, 9.514464, 9.523373, 9.532282, 9.541191, 9.5501  , 9.559009, 9.567918, 9.576827, 9.585736, 9.594645, 9.603554, 9.612462, 9.621371, 9.63028, 9.639189, 9.648098, 9.657007, 9.665916, 9.674825, 9.683734, 9.692643, 9.701552, 9.71046 , 9.719369, 9.728278, 9.737187, 9.746096, 9.755005, 9.763914, 9.772823, 9.781732, 9.790641, 9.79955 , 9.808458, 9.817367, 9.826276, 9.835185, 9.844094, 9.853003, 9.861912, 9.870821, 9.87973 , 9.888639, 9.897548, 9.906456, 9.915365, 9.924193, 9.932859, 9.941282, 9.949381, 9.957075, 9.964283, 9.970925, 9.976918]

  smoothed_nu_over_d_vals =  [0.001005, 0.001007, 0.001009, 0.001012, 0.001015, 0.001019, 0.001024, 0.00103, 0.001037, 0.001045, 0.001055, 0.001065, 0.001076, 0.001089, 0.001103, 0.001119, 0.001135, 0.001153, 0.001172, 0.001193, 0.001215, 0.001239, 0.001264, 0.00129, 0.001319, 0.001348, 0.00138 , 0.001413, 0.001448, 0.001485, 0.001523, 0.001564, 0.001607, 0.001652, 0.0017  , 0.00175 , 0.001802, 0.001857, 0.001915, 0.001976, 0.002041, 0.002108, 0.00218 , 0.002254, 0.002333, 0.002417, 0.002504, 0.002597, 0.002694, 0.002797, 0.002905, 0.00302 , 0.003141, 0.003268, 0.003403, 0.003546, 0.003697, 0.003856, 0.004024, 0.004202, 0.004391, 0.00459 , 0.004801, 0.005023, 0.005259, 0.005508, 0.005772, 0.006051, 0.006345, 0.006657, 0.006986, 0.007334, 0.007701, 0.008089, 0.008499, 0.008931, 0.009388, 0.009869, 0.010376, 0.010911, 0.011474, 0.012067, 0.012691, 0.013347, 0.014037, 0.014762, 0.015523, 0.016322, 0.017161, 0.01804 , 0.018961, 0.019926, 0.020935, 0.021991, 0.023095, 0.024247, 0.025451, 0.026707, 0.028016, 0.02938 , 0.0308  , 0.032278, 0.033814, 0.035411, 0.037069, 0.03879 , 0.040575, 0.042424, 0.044339, 0.046321, 0.048372, 0.050491, 0.052679, 0.054938, 0.057268, 0.059669, 0.062143, 0.06469 , 0.06731 , 0.070003, 0.07277 , 0.075611, 0.078526, 0.081515, 0.084578, 0.087714, 0.090925, 0.094208, 0.097564, 0.100992, 0.104492, 0.108063, 0.111705, 0.115416, 0.119195, 0.123042, 0.126956, 0.130935, 0.134979, 0.139086, 0.143254, 0.147483, 0.151771, 0.156116, 0.160517, 0.164973, 0.169482, 0.174041, 0.17865 , 0.183307, 0.18801 , 0.192756, 0.197545, 0.202374, 0.207242, 0.212147, 0.217086, 0.222058, 0.227062, 0.232094, 0.237153, 0.242238, 0.247346, 0.252475, 0.257624, 0.26279 , 0.267973, 0.273169, 0.278378, 0.283597, 0.288824, 0.294059, 0.299298, 0.304541, 0.309786, 0.315031, 0.320274, 0.325515, 0.330751, 0.335981, 0.341204, 0.346418, 0.351621, 0.356813, 0.361993, 0.367158, 0.372308, 0.377441, 0.382557, 0.387654, 0.392731, 0.397787, 0.402822, 0.407834, 0.412822, 0.417786, 0.422725, 0.427637, 0.432523, 0.437381, 0.442211, 0.447013, 0.451784, 0.456526, 0.461237, 0.465917, 0.470565, 0.475182, 0.479766, 0.484317, 0.488835, 0.49332 , 0.497771, 0.502187, 0.50657 , 0.510918, 0.515231, 0.519509, 0.523753, 0.527961, 0.532133, 0.536271, 0.540373, 0.544439, 0.54847 , 0.552465, 0.556425, 0.560349, 0.564238, 0.568091, 0.571908, 0.575691, 0.579438, 0.58315 , 0.586827, 0.590469, 0.594076, 0.597648, 0.601186, 0.60469, 0.60816 , 0.611595, 0.614997, 0.618365, 0.621699, 0.625001, 0.628269, 0.631505, 0.634708, 0.637878, 0.641017, 0.644124, 0.647199, 0.650242, 0.653255, 0.656237, 0.659188, 0.662108, 0.664999, 0.667859, 0.67069 , 0.673492, 0.676264, 0.679008, 0.681723, 0.68441 , 0.687069, 0.6897  , 0.692304, 0.69488 , 0.697429, 0.699952, 0.702448, 0.704919, 0.707363, 0.709782, 0.712175, 0.714543, 0.716886, 0.719205, 0.7215  , 0.72377 , 0.726017, 0.72824 , 0.730439, 0.732616, 0.73477 , 0.736901, 0.739011, 0.741098, 0.743163, 0.745207, 0.747229, 0.74923 , 0.751211, 0.753171, 0.75511 , 0.75703 , 0.758929, 0.760809, 0.762669, 0.76451 , 0.766332, 0.768136, 0.76992 , 0.771687, 0.773435, 0.775165, 0.776877, 0.778572, 0.78025 , 0.78191, 0.783554, 0.785181, 0.786791, 0.788385, 0.789963, 0.791524, 0.793071, 0.794601, 0.796116, 0.797616, 0.7991  , 0.80057 , 0.802025, 0.803466, 0.804892, 0.806304, 0.807702, 0.809087, 0.810457, 0.811814, 0.813157, 0.814488, 0.815805, 0.817109, 0.818401, 0.81968 , 0.820946, 0.8222  , 0.823442, 0.824672, 0.82589 , 0.827097, 0.828291, 0.829474, 0.830646, 0.831807, 0.832956, 0.834095, 0.835222, 0.836339, 0.837445, 0.838541, 0.839627, 0.840702, 0.841767, 0.842822, 0.843868, 0.844903, 0.845929, 0.846945, 0.847952, 0.84895 , 0.849938, 0.850917, 0.851888, 0.852849, 0.853801, 0.854745, 0.85568 , 0.856607, 0.857525, 0.858435, 0.859337, 0.86023, 0.861116, 0.861993, 0.862863, 0.863725, 0.864579, 0.865425, 0.866264, 0.867096, 0.86792 , 0.868737, 0.869547, 0.870349, 0.871145, 0.871934, 0.872715, 0.87349, 0.874258, 0.87502 , 0.875775, 0.876523, 0.877265, 0.878001, 0.87873 , 0.879453, 0.88017 , 0.880881, 0.881585, 0.882284, 0.882977, 0.883664, 0.884345, 0.88502, 0.88569 , 0.886354, 0.887013, 0.887666, 0.888314, 0.888956, 0.889593, 0.890225, 0.890852, 0.891473, 0.892089, 0.8927  , 0.893307, 0.893908, 0.894504, 0.895096, 0.895683, 0.896265, 0.896842, 0.897415, 0.897983, 0.898547, 0.899106, 0.89966, 0.900211, 0.900757, 0.901298, 0.901835, 0.902368, 0.902897, 0.903422, 0.903942, 0.904459, 0.904971, 0.90548 , 0.905984, 0.906485, 0.906982, 0.907475, 0.907964, 0.908449, 0.908931, 0.909409, 0.909883, 0.910354, 0.910821, 0.911285, 0.911745, 0.912202, 0.912655, 0.913105, 0.913551, 0.913994, 0.914434, 0.914871, 0.915304, 0.915734, 0.916161, 0.916585, 0.917005, 0.917423, 0.917837, 0.918249, 0.918657, 0.919062, 0.919465, 0.919865, 0.920261, 0.920655, 0.921046, 0.921434, 0.92182, 0.922202, 0.922582, 0.922959, 0.923334, 0.923706, 0.924075, 0.924441, 0.924805, 0.925167, 0.925526, 0.925882, 0.926236, 0.926587, 0.926936, 0.927283, 0.927627, 0.927969, 0.928308, 0.928645, 0.92898 , 0.929312, 0.929642, 0.92997 , 0.930296, 0.930619, 0.93094 , 0.931259, 0.931576, 0.931891, 0.932204, 0.932514, 0.932822, 0.933129, 0.933433, 0.933735, 0.934036, 0.934334, 0.93463 , 0.934924, 0.935217, 0.935507, 0.935796, 0.936082, 0.936367, 0.93665 , 0.936931, 0.93721 , 0.937488, 0.937763, 0.938037, 0.938309, 0.938579, 0.938848, 0.939115, 0.93938 , 0.939643, 0.939905, 0.940165, 0.940423, 0.94068 , 0.940935, 0.941189, 0.94144 , 0.941691, 0.941939, 0.942187, 0.942432, 0.942676, 0.942919, 0.94316 , 0.9434  , 0.943638, 0.943874, 0.944109, 0.944343, 0.944575, 0.944806, 0.945035, 0.945263, 0.94549, 0.945715, 0.945939, 0.946161, 0.946383, 0.946602, 0.946821, 0.947038, 0.947254, 0.947468, 0.947682, 0.947894, 0.948104, 0.948314, 0.948522, 0.948729, 0.948935, 0.949139, 0.949342, 0.949544, 0.949745, 0.949945, 0.950144, 0.950341, 0.950537, 0.950732, 0.950926, 0.951119, 0.951311, 0.951501, 0.951691, 0.951879, 0.952067, 0.952253, 0.952438, 0.952622, 0.952805, 0.952987, 0.953168, 0.953348, 0.953527, 0.953705, 0.953882, 0.954058, 0.954233, 0.954407, 0.95458 , 0.954752, 0.954923, 0.955093, 0.955262, 0.95543 , 0.955597, 0.955763, 0.955929, 0.956093, 0.956257, 0.956419, 0.956581, 0.956742, 0.956902, 0.957061, 0.957219, 0.957377, 0.957533, 0.957689, 0.957844, 0.957998, 0.958151, 0.958303, 0.958455, 0.958605, 0.958755, 0.958904, 0.959053, 0.9592  , 0.959347, 0.959493, 0.959638, 0.959782, 0.959926, 0.960069, 0.960211, 0.960352, 0.960493, 0.960633, 0.960772, 0.96091 , 0.961048, 0.961185, 0.961321, 0.961457, 0.961591, 0.961725, 0.961859, 0.961992, 0.962124, 0.962255, 0.962386, 0.962516, 0.962645, 0.962774, 0.962902, 0.963029, 0.963156, 0.963282, 0.963408, 0.963532, 0.963657, 0.96378 , 0.963903, 0.964025, 0.964147, 0.964268, 0.964389, 0.964509, 0.964628, 0.964747, 0.964865, 0.964982, 0.965099, 0.965215, 0.965331, 0.965446, 0.965561, 0.965675, 0.965789, 0.965901, 0.966014, 0.966126, 0.966237, 0.966348, 0.966458, 0.966568, 0.966677, 0.966785, 0.966894, 0.967001, 0.967108, 0.967215, 0.967321, 0.967426, 0.967531, 0.967636, 0.96774, 0.967843, 0.967946, 0.968049, 0.968151, 0.968252, 0.968353, 0.968454, 0.968554, 0.968654, 0.968753, 0.968852, 0.96895 , 0.969048, 0.969145, 0.969242, 0.969338, 0.969434, 0.96953 , 0.969625, 0.969719, 0.969814, 0.969907, 0.970001, 0.970094, 0.970186, 0.970278, 0.97037 , 0.970461, 0.970552, 0.970642, 0.970732, 0.970822, 0.970911, 0.971   , 0.971088, 0.971176, 0.971263, 0.971351, 0.971437, 0.971524, 0.97161 , 0.971695, 0.97178 , 0.971865, 0.97195 , 0.972034, 0.972117, 0.972201, 0.972284, 0.972366, 0.972448, 0.97253 , 0.972612, 0.972693, 0.972774, 0.972854, 0.972934, 0.973014, 0.973093, 0.973172, 0.973251, 0.973329, 0.973407, 0.973485, 0.973562, 0.973639, 0.973716, 0.973792, 0.973868, 0.973944, 0.974019, 0.974094, 0.974169, 0.974243, 0.974317, 0.974391, 0.974464, 0.974537, 0.97461 , 0.974682, 0.974755, 0.974826, 0.974898, 0.974969, 0.97504 , 0.975111, 0.975181, 0.975251, 0.975321, 0.97539 , 0.97546 , 0.975528, 0.975597, 0.975665, 0.975733, 0.975801, 0.975869, 0.975936, 0.976003, 0.976069, 0.976136, 0.976202, 0.976268, 0.976333, 0.976398, 0.976463, 0.976528, 0.976592, 0.976657, 0.976721, 0.976784, 0.976848, 0.976911, 0.976974, 0.977036, 0.977099, 0.977161, 0.977223, 0.977284, 0.977346, 0.977407, 0.977468, 0.977529, 0.977589, 0.977649, 0.977709, 0.977769, 0.977828, 0.977887, 0.977946, 0.978005, 0.978064, 0.978122, 0.97818 , 0.978238, 0.978295, 0.978353, 0.97841 , 0.978467, 0.978523, 0.97858 , 0.978636, 0.978692, 0.978748, 0.978803, 0.978859, 0.978914, 0.978969, 0.979024, 0.979078, 0.979132, 0.979187, 0.97924 , 0.979294, 0.979348, 0.979401, 0.979454, 0.979507, 0.979559, 0.979612, 0.979664, 0.979716, 0.979768, 0.97982 , 0.979871, 0.979922, 0.979973, 0.980024, 0.980075, 0.980125, 0.980176, 0.980226, 0.980276, 0.980325, 0.980375, 0.980424, 0.980474, 0.980523, 0.980571, 0.98062 , 0.980668, 0.980717, 0.980765, 0.980813, 0.98086 , 0.980908, 0.980955, 0.981002, 0.981049, 0.981096, 0.981143, 0.981189, 0.981236, 0.981282, 0.981328, 0.981374, 0.981419, 0.981465, 0.98151 , 0.981555, 0.9816  , 0.981645, 0.98169 , 0.981734, 0.981779, 0.981823, 0.981867, 0.981911, 0.981954, 0.981998, 0.982041, 0.982085, 0.982128, 0.982171, 0.982213, 0.982256, 0.982298, 0.982341, 0.982383, 0.982425, 0.982467, 0.982508, 0.98255 , 0.982591, 0.982633, 0.982674, 0.982715, 0.982756, 0.982796, 0.982837, 0.982877, 0.982918, 0.982958, 0.982998, 0.983038, 0.983077, 0.983117, 0.983156, 0.983195, 0.983235, 0.983274, 0.983313, 0.983351, 0.98339 , 0.983428, 0.983467, 0.983505, 0.983543, 0.983581, 0.983619, 0.983656, 0.983694, 0.983731, 0.983769, 0.983806, 0.983843, 0.98388 , 0.983917, 0.983953, 0.98399 , 0.984026, 0.984063, 0.984099, 0.984135, 0.984171, 0.984206, 0.984242, 0.984278, 0.984313, 0.984348, 0.984384, 0.984419, 0.984454, 0.984489, 0.984523, 0.984558, 0.984592, 0.984627, 0.984661, 0.984695, 0.984729, 0.984763, 0.984797, 0.984831, 0.984864, 0.984898, 0.984931, 0.984964, 0.984997, 0.98503 , 0.985061, 0.985091, 0.985119, 0.985145, 0.98517 , 0.985192]

  return smoothed_alpha_vals, smoothed_nu_over_d_vals
  
  
def smooth_this_array(input_array : np.ndarray, 
                      smoothing_window_length : int = 10, 
                      num_decimal_places : int = 3) -> np.ndarray:

  len_input_array = input_array.shape[0]
  smoothing_parameters = np.ones(smoothing_window_length)
  for i in range(smoothing_window_length):
    smoothing_parameters[i] = min(i + 1, smoothing_window_length - i)
  print(smoothing_parameters)
  smoothing_parameters = smoothing_parameters / np.sum(smoothing_parameters)
  print(smoothing_parameters)

  len_input_array = input_array.shape[0]
  len_extended_array = len_input_array + 2 * (smoothing_window_length - 1)

  extended_input_array = np.zeros(len_extended_array)
  for i in range(len_extended_array):
    this_index = max(0, min(i - smoothing_window_length + 1, len_input_array - 1))
    extended_input_array[i] = input_array[this_index]

  smoothed_values_initial = np.convolve(extended_input_array, smoothing_parameters, mode='same')
  smoothed_values_final = smoothed_values_initial[smoothing_window_length - 2 : -smoothing_window_length]

  dp_factor = 10**num_decimal_places
  for i in range(len(smoothed_values_final)):
    smoothed_values_final[i] = round(smoothed_values_final[i] * dp_factor) / dp_factor

  return smoothed_values_final
  
def calculate_softmax_alpha(alpha : float) -> float:
    
  # ** TO DO : use new alpha
  # ** TO DO : adapt to GPU?
  
  print(" ** calculate_softmax_alpha : need to re-do hardcoded values ** ")

  alpha_vals, softmax_alpha_vals = get_alpha_vals_softmax_alpha_vals()

  if alpha < alpha_vals[0]:
    smallest_legitimate_alpha = 1.0
    default_value = 1.0
    smallest_alpha_val = alpha_vals[0]
    smallest_softmax_alpha_val = softmax_alpha_vals[0]
    if alpha < smallest_legitimate_alpha:
      return default_value
    else:
      gradient = ((smallest_softmax_alpha_val - default_value) / (smallest_alpha_val - smallest_legitimate_alpha))
      interpolated_value = gradient * (alpha - smallest_legitimate_alpha) + default_value
      return interpolated_value
  if alpha > alpha_vals[-1]:
    return alpha

  for i in range(len(alpha_vals)):
    if alpha < alpha_vals[i]:
      if i == len(alpha_vals) - 1:
        return alpha
      # do some interpolation:
      t = (alpha - alpha_vals[i - 1]) / (alpha_vals[i] - alpha_vals[i - 1])
      interpolated_value = (1 - t) * softmax_alpha_vals[i - 1] + t * softmax_alpha_vals[i]
      return interpolated_value
      
def get_alpha_vals_softmax_alpha_vals_old() -> tuple:
    
  print("*** get_alpha_vals_softmax_alpha_vals : deprecate these hard-coded values? ***")

  alpha_vals =  [1.1, 1.11, 1.12, 1.130, 1.140, 1.150, 1.160, 1.170, 1.180, 1.190, 1.20, 1.210, 1.220, 1.23, 1.240, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.380, 1.390, 1.40, 1.410, 1.420, 1.430, 1.440, 1.450, 1.46, 1.470, 1.48, 1.490, 1.5, 1.510, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.630, 1.640, 1.650, 1.660, 1.670, 1.680, 1.69, 1.70, 1.71, 1.720, 1.73, 1.740, 1.75, 1.760, 1.77, 1.780, 1.79, 1.80, 1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.880, 1.890, 1.90, 1.910, 1.920, 1.930, 1.94, 1.950, 1.96, 1.970, 1.98, 1.990, 2.0, 2.010, 2.02, 2.030, 2.04, 2.050, 2.06, 2.070, 2.08, 2.09]

  raw_softmax_alpha_vals =  [1.2247247247247248, 1.3761761761761764, 1.4830830830830832, 1.518718718718719, 1.598898898898899, 1.518718718718719, 1.4830830830830832, 1.5454454454454456, 1.5632632632632635, 1.7058058058058059, 1.6790790790790793, 1.6701701701701703, 1.7503503503503506, 1.6701701701701703, 1.6612612612612614, 1.7058058058058059, 1.7058058058058059, 1.7503503503503506, 1.7414414414414416, 1.7325325325325327, 1.8216216216216217, 1.8572572572572574, 1.7325325325325327, 1.7681681681681685, 1.8394394394394395, 1.8661661661661664, 1.8572572572572574, 1.8661661661661664, 1.8661661661661664, 1.8928928928928932, 1.9196196196196198, 1.8661661661661664, 1.8839839839839843, 1.9730730730730732, 1.9374374374374377, 1.9107107107107109, 1.9196196196196198, 1.9463463463463464, 1.8483483483483485, 1.9463463463463464, 1.9196196196196198, 1.9641641641641643, 1.9463463463463464, 1.9018018018018021, 1.9819819819819822, 1.9641641641641643, 1.8928928928928932, 1.9018018018018021, 1.8928928928928932, 1.8661661661661664, 1.9285285285285287, 1.8216216216216217, 1.9018018018018021, 1.9374374374374377, 1.8572572572572574, 1.8305305305305306, 1.8572572572572574, 1.9285285285285287, 1.8928928928928932, 1.9196196196196198, 1.8572572572572574, 1.8928928928928932, 1.8839839839839843, 1.8750750750750753, 1.8750750750750753, 1.8661661661661664, 1.8661661661661664, 1.8661661661661664, 1.8572572572572574, 1.8661661661661664, 1.8661661661661664, 1.8661661661661664, 1.8661661661661664, 1.8750750750750753, 1.8750750750750753, 1.8839839839839843, 1.8928928928928932, 1.9018018018018021, 1.9018018018018021, 1.9107107107107109, 1.9196196196196198, 1.9285285285285287, 1.9374374374374377, 1.9463463463463464, 1.9552552552552553, 1.9641641641641643, 1.9730730730730732, 1.9819819819819822, 1.990890890890891, 1.9997997997998, 2.008708708708709, 2.017617617617618, 2.026526526526527, 2.035435435435436, 2.0443443443443448, 2.0532532532532537, 2.071071071071071, 2.07997997997998, 2.088888888888889, 2.097797797797798]

  smoothed_softmax_alpha_vals = [1.31187005, 1.33705797, 1.36637638, 1.39836746, 1.43141141, 1.46348348, 1.49344981, 1.52171535, 1.54933297, 1.57646465, 1.60230048, 1.62530167, 1.64522523, 1.66271908, 1.67737829, 1.69049868, 1.70386204, 1.7181163, 1.73204659, 1.74500501, 1.75723451, 1.76930203, 1.78201747, 1.79351806, 1.80469469, 1.81789608, 1.83109746, 1.84308399, 1.85417963, 1.86462735, 1.87386022, 1.88244517, 1.8895723 , 1.89669943, 1.90431249, 1.90990081, 1.9140313 , 1.91775685, 1.92083447, 1.92294021, 1.92472199, 1.92593685, 1.92674675, 1.92626081, 1.92447902, 1.92245427, 1.91994358, 1.91581309, 1.91087269, 1.90641824, 1.90236873, 1.89840022, 1.89451269, 1.8911921, 1.88900537, 1.88665666, 1.88414596, 1.88309309, 1.88333606, 1.88333606, 1.88276913, 1.88171626, 1.88025844, 1.87920557, 1.8773428 , 1.8756420, 1.87450814, 1.87337428, 1.87215943, 1.87118755, 1.87118755, 1.87191646, 1.87361725, 1.87580399, 1.87888161, 1.88276913, 1.88738557, 1.89273091, 1.89864319, 1.90504141, 1.91192556, 1.91921467, 1.92690873, 1.93492674, 1.94318773, 1.9516107 , 1.96019565, 1.96886159, 1.97768951, 1.98667941, 1.99583129, 2.00506416, 2.01429702, 2.0234489 , 2.0324388 , 2.04118573, 2.0496087 , 2.05762672, 2.06507781, 2.07179998]

  return alpha_vals, smoothed_softmax_alpha_vals
  
def get_alpha_vals_softmax_alpha_vals() -> tuple:
    
  print("** get_alpha_vals_softmax_alpha_vals : need to recalculate for new alpha **")
    
  # ** TO DO : recalculate for new alpha
    
  # recalculated, with d = 1000

  alpha_vals =  [1.292893, 1.292893, 1.292893, 1.292894, 1.292897, 1.292911, 1.292955, 1.293055, 1.293245, 1.293557, 1.294038, 1.294684, 1.295543, 1.296625, 1.29794, 1.299504, 1.301304, 1.303359, 1.305674, 1.30818, 1.310943, 1.313959, 1.317188, 1.320625, 1.324296, 1.328177, 1.332258, 1.336529, 1.340988, 1.34566, 1.350474, 1.355493, 1.36074, 1.366138, 1.371566, 1.377244, 1.383176, 1.389214, 1.395392, 1.401758, 1.408105, 1.414957, 1.421658, 1.428517, 1.435682, 1.44288, 1.450256, 1.457864, 1.465347, 1.473098, 1.480941, 1.489303, 1.497161, 1.50557, 1.514234, 1.522268, 1.53083, 1.539429, 1.548303, 1.557392, 1.566224, 1.575139, 1.584542, 1.593386, 1.602557, 1.611905, 1.621286, 1.630603, 1.640193, 1.649601, 1.659092, 1.668721, 1.678427, 1.688029, 1.69767, 1.707359, 1.717062, 1.727011, 1.736628, 1.746348, 1.756148, 1.765976, 1.775763, 1.785947, 1.795428, 1.805268, 1.815131, 1.825009, 1.835023, 1.844704, 1.85459, 1.864619, 1.874391, 1.884239, 1.894139, 1.90406, 1.914272, 1.923841, 1.933891, 1.943743, 1.953564, 1.963465, 1.973343, 1.983339, 1.993174, 2.003159, 2.013044, 2.023106, 2.032828, 2.04332, 2.05269, 2.06262, 2.072651, 2.082727, 2.092496, 2.102354, 2.112365, 2.12214, 2.132094, 2.1424, 2.151961, 2.161999, 2.171999, 2.181854, 2.192106, 2.201747, 2.211629, 2.221516, 2.231549, 2.241479, 2.251745, 2.261446, 2.27138, 2.281283, 2.29184, 2.301444, 2.311139, 2.321141, 2.33107, 2.340925, 2.350903, 2.36081, 2.370808, 2.381072, 2.390697, 2.400742, 2.410749, 2.420992, 2.430669, 2.440715, 2.450659, 2.460663, 2.471043, 2.480436, 2.490421, 2.500754, 2.510341, 2.520477, 2.530428, 2.540407, 2.550483, 2.560484, 2.570312, 2.580127, 2.590095, 2.600188, 2.610322, 2.620473, 2.63057, 2.640249, 2.650178, 2.659955, 2.670189, 2.680027, 2.69045, 2.700059, 2.71023, 2.720053, 2.729786, 2.740007, 2.749726, 2.760061, 2.769698, 2.779888, 2.789871, 2.800134, 2.809785, 2.81996, 2.830041, 2.839753, 2.849761, 2.859562, 2.87007, 2.87968, 2.889508, 2.899799, 2.909613, 2.919539, 2.929829]

  smoothed_softmax_alpha_vals =  [1.292914, 1.292996, 1.293203, 1.293678, 1.294728, 1.296808, 1.300562, 1.306794, 1.316385, 1.329934, 1.347666, 1.369349, 1.394165, 1.420706, 1.447452, 1.473202, 1.496971, 1.518262, 1.537178, 1.554416, 1.570556, 1.586047, 1.601366, 1.616842, 1.632358, 1.647563, 1.662246, 1.676216, 1.689213, 1.700968, 1.711585, 1.721368, 1.730587, 1.739492, 1.74844, 1.757716, 1.767471, 1.777674, 1.788309, 1.799396, 1.810806, 1.822261, 1.833365, 1.843837, 1.853305, 1.861474, 1.868283, 1.874073, 1.879167, 1.883929, 1.888895, 1.894462, 1.900665, 1.907513, 1.915122, 1.923224, 1.931464, 1.939668, 1.947755, 1.955318, 1.962191, 1.968444, 1.974084, 1.978888, 1.982925, 1.986353, 1.989125, 1.991371, 1.993437, 1.995663, 1.998161, 2.001109, 2.004468, 2.007972, 2.011216, 2.014023, 2.01624, 2.017922, 2.01929, 2.020568, 2.021905, 2.023332, 2.024745, 2.025922, 2.026976, 2.028079, 2.029484, 2.031391, 2.034022, 2.037213, 2.040511, 2.043518, 2.045851, 2.047506, 2.0488, 2.050121, 2.051793, 2.054179, 2.057428, 2.06107, 2.064822, 2.068867, 2.073481, 2.078871, 2.085619, 2.094453, 2.10529, 2.117694, 2.130942, 2.144309, 2.156784, 2.167711, 2.176881, 2.184373, 2.190499, 2.195594, 2.200096, 2.204276, 2.208387, 2.212503, 2.216734, 2.221157, 2.225794, 2.230647, 2.23571, 2.240998, 2.246485, 2.252183, 2.258134, 2.264367, 2.270894, 2.277711, 2.284805, 2.292135, 2.299655, 2.307329, 2.315131, 2.323038, 2.331039, 2.339133, 2.347324, 2.355613, 2.364003, 2.372495, 2.381086, 2.389773, 2.398551, 2.407415, 2.416359, 2.42538, 2.434474, 2.443635, 2.452856, 2.462131, 2.471452, 2.480815, 2.490213, 2.499648, 2.509122, 2.518631, 2.528174, 2.537745, 2.547337, 2.556942, 2.566553, 2.576169, 2.585792, 2.595427, 2.605077, 2.614749, 2.624444, 2.634164, 2.643907, 2.653672, 2.663454, 2.673248, 2.683049, 2.692851, 2.702649, 2.712444, 2.722236, 2.73203, 2.741834, 2.751652, 2.761489, 2.771348, 2.781226, 2.791119, 2.80102, 2.810922, 2.820819, 2.830708, 2.840588, 2.850455, 2.860281, 2.870011, 2.879314, 2.887863, 2.895337]
  
  new_alpha_vals = np.array(alpha_vals) - 1.0
  new_smoothed_softmax_alpha_vals  = np.array(smoothed_softmax_alpha_vals ) - 1.0

  return new_alpha_vals, new_smoothed_softmax_alpha_vals
  #return alpha_vals, smoothed_softmax_alpha_vals  
  
def attention_experiment_new(N : int,
                         d : int,
                         alpha_X : float,
                         alpha_Q : float,
                         alpha_K : float,
                         alpha_V : float,
                         verbose : bool = False
                         ) -> tuple:
                             
  # uses correct alpha
  # ** TO DO : adapt to GPU?                        

  # N : number of datapoints (batch size)
  # d : ambient (model) dimension
  # alpha_X : tail exponent of data manifold X (N * d matrix)
  # alpha_Q : tail exponent of WQ (d * d matrix)
  # alpha_K : tail exponent of WK (d * d matrix)
  # alpha_V : tail exponent of WV (d * d matrix)

  X = generate_data_manifold(N, d, alpha_X)
  WQ = generate_square_weight_matrix(d, alpha_Q)
  WK = generate_square_weight_matrix(d, alpha_K)
  WV = generate_square_weight_matrix(d, alpha_V)
  
  dim_X = X.shape[1]
  dim_WQ = WQ.shape[1]
  dim_WK = WK.shape[1]
  dim_WV = WV.shape[1]

  pp_dim_X = calculate_PatnaikPearson_dim(X)
  pp_dim_WQ = calculate_PatnaikPearson_dim(WQ)
  pp_dim_WK = calculate_PatnaikPearson_dim(WK)
  pp_dim_WV = calculate_PatnaikPearson_dim(WV)

  implied_alpha_X = calculate_alpha_given_nu_over_d_and_d(pp_dim_X / dim_X, dim_X)
  implied_alpha_Q = calculate_alpha_given_nu_over_d_and_d(pp_dim_WQ / dim_WQ, dim_WQ)
  implied_alpha_K = calculate_alpha_given_nu_over_d_and_d(pp_dim_WK / dim_WK, dim_WK)
  implied_alpha_V = calculate_alpha_given_nu_over_d_and_d(pp_dim_WV / dim_WV, dim_WV)

  if verbose:
    print("alpha_X = ", alpha_X, ", pp_dim_X = ", pp_dim_X, ", implied_alpha_X = ", implied_alpha_X)
    print("alpha_Q = ", alpha_Q, ", pp_dim_WQ = ", pp_dim_WQ, ", implied_alpha_Q = ", implied_alpha_Q)
    print("alpha_K = ", alpha_K, ", pp_dim_WK = ", pp_dim_WK, ", implied_alpha_K = ", implied_alpha_K)
    print("alpha_V = ", alpha_V, ", pp_dim_WV = ", pp_dim_WV, ", implied_alpha_V = ", implied_alpha_V)

  XWQ = X @ WQ
  dim_XWQ = XWQ.shape[1]
  actual_pp_dim_XWQ = calculate_PatnaikPearson_dim(XWQ)
  estimate_pp_dim_XWQ = dim_XWQ * estimate_product_nu_over_d(pp_dim_X / dim_XWQ, dim_XWQ, pp_dim_WQ / dim_WQ, dim_WQ)
  actual_alpha_XWQ = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_XWQ / dim_XWQ, dim_XWQ)
  estimate_alpha_XWQ = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_Q, dim_WQ)
  
  if verbose:
    print("pp_dim_X = ", pp_dim_X, ", pp_dim_WQ = ", pp_dim_WQ, ", actual_pp_dim_XWQ = ", actual_pp_dim_XWQ, ", estimate_pp_dim_XWQ = ", estimate_pp_dim_XWQ)
    print("actual_alpha_XWQ = ", actual_alpha_XWQ, ", estimate_alpha_XWQ = ", estimate_alpha_XWQ)

  XWK = X @ WK
  dim_XWK = XWK.shape[1]
  actual_pp_dim_XWK = calculate_PatnaikPearson_dim(XWK)
  estimate_pp_dim_XWK = dim_XWK * estimate_product_nu_over_d(pp_dim_X / dim_X, dim_X, pp_dim_WK / dim_WK, dim_WK)
  actual_alpha_XWK = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_XWK / dim_XWK, dim_XWK)
  estimate_alpha_XWK = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_K, dim_WK)
  
  if verbose:
    print("pp_dim_X = ", pp_dim_X, ", pp_dim_WK = ", pp_dim_WK, ", actual_pp_dim_XWK = ", actual_pp_dim_XWK, ", estimate_pp_dim_XWK = ", estimate_pp_dim_XWK)
    print("actual_alpha_XWK = ", actual_alpha_XWK, ", estimate_alpha_XWK = ", estimate_alpha_XWK)

  XWV = X @ WV
  dim_XWV = XWV.shape[1]
  actual_pp_dim_XWV = calculate_PatnaikPearson_dim(XWV)
  estimate_pp_dim_XWV = dim_XWV * estimate_product_nu_over_d(pp_dim_X / dim_X, d, pp_dim_WV / dim_WV, dim_WV)
  actual_alpha_XWV = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_XWV / dim_XWV, dim_XWV)
  estimate_alpha_XWV = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_V, dim_WV)
  
  if verbose:
    print("pp_dim_X = ", pp_dim_X, ", pp_dim_WV = ", pp_dim_WV, ", actual_pp_dim_XWV = ", actual_pp_dim_XWV, ", estimate_pp_dim_XWV = ", estimate_pp_dim_XWV)
    print("actual_alpha_XWV = ", actual_alpha_XWV, ", estimate_alpha_XWV = ", estimate_alpha_XWV)

  QKT = XWQ @ XWK.T
  dim_QKT = QKT.shape[1]
  actual_pp_dim_QKT = calculate_PatnaikPearson_dim(QKT)
  estimate_pp_dim_QKT = dim_QKT * estimate_product_nu_over_d(actual_pp_dim_XWQ / dim_QKT, dim_QKT, actual_pp_dim_XWK / dim_XWK, dim_XWK)
  actual_alpha_QKT = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_QKT / dim_QKT, dim_QKT)
  estimate_alpha_QKT = estimate_product_alpha(implied_alpha_Q, dim_WQ, implied_alpha_K, dim_WK)
  
  if verbose:
    print("pp_dim_WQ = ", pp_dim_WQ, ", pp_dim_WK = ", pp_dim_WK, "actual_pp_dim_QKT = ", actual_pp_dim_QKT, ", estimate_pp_dim_QKT = ", estimate_pp_dim_QKT)
    print("alpha_Q = ", alpha_Q, ", alpha_K = ", alpha_K, ", actual_alpha_QKT = ", actual_alpha_QKT, "estimate_alpha_QKT = ", estimate_alpha_QKT)

  QKTrescaled = QKT / np.sqrt(d)
  dim_QKTrescaled = QKTrescaled.shape[1]
  actual_pp_dim_QKTrescaled = calculate_PatnaikPearson_dim(QKTrescaled)
  actual_alpha_QKTrescaled = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_QKTrescaled / dim_QKTrescaled, dim_QKTrescaled)
  
  if verbose:
    print("actual_pp_dim_QKT = ", actual_pp_dim_QKT, ", actual_pp_dim_QKTrescaled = ", actual_pp_dim_QKTrescaled)
    print("actual_alpha_QKT = ", actual_alpha_QKT, ",actual_alpha_QKTrescaled = ", actual_alpha_QKTrescaled)

  AttnQK = row_wise_softmax(QKTrescaled)
  dim_AttnQK = AttnQK.shape[1]
  actual_pp_dim_AttnQK = calculate_PatnaikPearson_dim(AttnQK)
  actual_alpha_AttnQK = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_AttnQK / dim_AttnQK, dim_AttnQK)
  estimate_alpha_AttnQK = calculate_softmax_alpha(actual_alpha_QKTrescaled)
  
  if verbose:
    print("sanity check : N = ", N, ", AttnQK.sum() = ", AttnQK.sum())
    print("actual_pp_dim_AttnQK = ", actual_pp_dim_AttnQK, ", actual_pp_dim_QKT = ", actual_pp_dim_QKT, ", actual_pp_dim_QKTrescaled = ", actual_pp_dim_QKTrescaled)
    print("actual_alpha_AttnQK = ", actual_alpha_AttnQK, ", estimate_alpha_AttnQK = ", estimate_alpha_AttnQK)

  estimate_pp_dim_AttnQK = dim_AttnQK * calculate_nu_over_d_given_alpha_and_d(estimate_alpha_AttnQK, dim_AttnQK)

  AttnQKV = AttnQK @ XWV
  dim_AttnQKV = AttnQKV.shape[1]

  actual_pp_dim_AttnQKV = (calculate_PatnaikPearson_dim(AttnQKV)).astype(float)
  actual_alpha_AttnQKV = calculate_alpha_given_nu_over_d_and_d(actual_pp_dim_AttnQKV / dim_AttnQKV, dim_AttnQKV)

  estimate_alpha_XWQ = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_Q, dim_WQ)
  estimate_alpha_XWK = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_K, dim_WK)
  estimate_alpha_QKT = estimate_product_alpha(estimate_alpha_XWQ, dim_XWQ, estimate_alpha_XWK, dim_XWK)
  estimate_alpha_softmax_QKT = calculate_softmax_alpha(estimate_alpha_QKT)
  estimate_alpha_XWV = estimate_product_alpha(implied_alpha_X, dim_X, implied_alpha_V, dim_WV)
  estimate_alpha_AttnQKV = estimate_product_alpha(estimate_alpha_softmax_QKT, dim_QKT, estimate_alpha_XWV, dim_XWV)
  estimate_pp_dim_AttnQKV = dim_AttnQKV * calculate_nu_over_d_given_alpha_and_d(estimate_alpha_AttnQKV, dim_AttnQKV)

  if verbose:
    print("actual_alpha_AttnQKV = ", actual_alpha_AttnQKV, ", estimate_alpha_AttnQKV = ", estimate_alpha_AttnQKV)
    print("actual_pp_dim_AttnQKV = ", actual_pp_dim_AttnQKV, ", estimate_pp_dim_AttnQKV = ", estimate_pp_dim_AttnQKV)
    
  results_dict = {
    "actual_pp_dim_AttnQKV" : actual_pp_dim_AttnQKV, 
    "estimate_pp_dim_AttnQKV" :  estimate_pp_dim_AttnQKV,
    "actual_alpha_AttnQKV" : actual_alpha_AttnQKV,
    "estimate_alpha_AttnQKV" : estimate_alpha_AttnQKV
  }

  return results_dict

def attention_experiment_old(N : int, 
                         d : int, 
                         alpha_X : float, 
                         alpha_Q : float, 
                         alpha_K : float, 
                         alpha_V : float,
                         verbose : bool = False
                         ) -> tuple:
                             
  # ** TO DO : deprecate

  # N : number of datapoints (batch size)
  # d : ambient (model) dimension
  # alpha_X : tail exponent of data manifold X (N * d matrix)
  # alpha_Q : tail exponent of WQ (d * d matrix)
  # alpha_K : tail exponent of WK (d * d matrix)
  # alpha_V : tail exponent of WV (d * d matrix)

  X = generate_data_manifold(N, d, alpha_X)
  WQ = generate_square_weight_matrix(d, alpha_Q)
  WK = generate_square_weight_matrix(d, alpha_K)
  WV = generate_square_weight_matrix(d, alpha_V)

  nu_X = calculate_PatnaikPearson_dim(X)
  nu_WQ = calculate_PatnaikPearson_dim(WQ)
  nu_WK = calculate_PatnaikPearson_dim(WK)
  nu_WV = calculate_PatnaikPearson_dim(WV)

  implied_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_X / d, d)
  implied_alpha_Q = calculate_alpha_given_nu_over_d_and_d(nu_WQ / d, d)
  implied_alpha_K = calculate_alpha_given_nu_over_d_and_d(nu_WK / d, d)
  implied_alpha_V = calculate_alpha_given_nu_over_d_and_d(nu_WV / d, d)

  if verbose:
    print("alpha_X = ", alpha_X, ", nu_X = ", nu_X, ", implied_alpha_X = ", implied_alpha_X)
    print("alpha_Q = ", alpha_Q, ", nu_WQ = ", nu_WQ, ", implied_alpha_Q = ", implied_alpha_Q)
    print("alpha_K = ", alpha_K, ", nu_WK = ", nu_WK, ", implied_alpha_K = ", implied_alpha_K)
    print("alpha_V = ", alpha_V, ", nu_WV = ", nu_WV, ", implied_alpha_V = ", implied_alpha_V)

  XWQ = X @ WQ
  actual_nu_XWQ = calculate_PatnaikPearson_dim(XWQ)
  estimate_nu_XWQ = d * estimate_product_nu_over_d(nu_X / d, d, nu_WQ / d, d)
  actual_alpha_XWQ = calculate_alpha_given_nu_over_d_and_d(actual_nu_XWQ / d, d)
  estimate_alpha_XWQ = estimate_product_alpha(alpha_X, d, alpha_Q, d)
  if verbose:
    print("nu_X = ", nu_X, ", nu_WQ = ", nu_WQ, ", actual_nu_XWQ = ", actual_nu_XWQ, ", estimate_nu_XWQ = ", estimate_nu_XWQ)
    print("actual_alpha_XWQ = ", actual_alpha_XWQ, ", estimate_alpha_XWQ = ", estimate_alpha_XWQ)

  XWK = X @ WK
  actual_nu_XWK = calculate_PatnaikPearson_dim(XWK)
  estimate_nu_XWK = d * estimate_product_nu_over_d(nu_X / d, d, nu_WK / d, d)
  actual_alpha_XWK = calculate_alpha_given_nu_over_d_and_d(actual_nu_XWK / d, d)
  estimate_alpha_XWK = estimate_product_alpha(alpha_X, d, alpha_K, d)
  if verbose:
    print("nu_X = ", nu_X, ", nu_WK = ", nu_WK, ", actual_nu_XWK = ", actual_nu_XWK, ", estimate_nu_XWK = ", estimate_nu_XWK)
    print("actual_alpha_XWK = ", actual_alpha_XWK, ", estimate_alpha_XWK = ", estimate_alpha_XWK)

  XWV = X @ WV
  actual_nu_XWV = calculate_PatnaikPearson_dim(XWV)
  estimate_nu_XWV = d * estimate_product_nu_over_d(nu_X / d, d, nu_WV / d, d)
  actual_alpha_XWV = calculate_alpha_given_nu_over_d_and_d(actual_nu_XWV / d, d)
  estimate_alpha_XWV = estimate_product_alpha(alpha_X, d, alpha_V, d)
  if verbose:
    print("nu_X = ", nu_X, ", nu_WV = ", nu_WV, ", actual_nu_XWV = ", actual_nu_XWV, ", estimate_nu_XWV = ", estimate_nu_XWV)
    print("actual_alpha_XWV = ", actual_alpha_XWV, ", estimate_alpha_XWV = ", estimate_alpha_XWV)

  QKT = XWQ @ XWK.T
  actual_nu_QKT = calculate_PatnaikPearson_dim(QKT)
  estimate_nu_QKT = d * estimate_product_nu_over_d(actual_nu_XWQ / d, d, actual_nu_XWK / d, d)
  actual_alpha_QKT = calculate_alpha_given_nu_over_d_and_d(actual_nu_QKT / d, d)
  estimate_alpha_QKT = estimate_product_alpha(alpha_Q, d, alpha_K, d)
  if verbose: 
    print("nu_WQ = ", nu_WQ, ", nu_WK = ", nu_WK, "actual_nu_QKT = ", actual_nu_QKT, ", estimate_nu_QKT = ", estimate_nu_QKT)
    print("alpha_Q = ", alpha_Q, ", alpha_K = ", alpha_K, ", actual_alpha_QKT = ", actual_alpha_QKT, "estimate_alpha_QKT = ", estimate_alpha_QKT)

  QKTrescaled = QKT / np.sqrt(d)
  actual_nu_QKTrescaled = calculate_PatnaikPearson_dim(QKTrescaled)
  actual_alpha_QKTrescaled = calculate_alpha_given_nu_over_d_and_d(actual_nu_QKTrescaled / d, d)
  if verbose:
    print("actual_nu_QKT = ", actual_nu_QKT, ", actual_nu_QKTrescaled = ", actual_nu_QKTrescaled)
    print("actual_alpha_QKT = ", actual_alpha_QKT, ",actual_alpha_QKTrescaled = ", actual_alpha_QKTrescaled)

  AttnQK = row_wise_softmax(QKTrescaled)
  actual_nu_AttnQK = calculate_PatnaikPearson_dim(AttnQK)
  actual_alpha_AttnQK = calculate_alpha_given_nu_over_d_and_d(actual_nu_AttnQK / d, d)
  estimate_alpha_AttnQK = calculate_softmax_alpha(actual_alpha_QKTrescaled)
  if verbose:
    print("sanity check : N = ", N, ", AttnQK.sum() = ", AttnQK.sum())
    print("actual_nu_AttnQK = ", actual_nu_AttnQK, ", actual_nu_QKT = ", actual_nu_QKT, ", actual_nu_QKTrescaled = ", actual_nu_QKTrescaled)
    print("actual_alpha_AttnQK = ", actual_alpha_AttnQK, ", estimate_alpha_AttnQK = ", estimate_alpha_AttnQK)

  estimate_nu_AttnQK = d * calculate_nu_over_d_given_alpha_and_d(estimate_alpha_AttnQK, d)

  AttnQKV = AttnQK @ XWV

  actual_nu_AttnQKV = (calculate_PatnaikPearson_dim(AttnQKV)).astype(float)
  actual_alpha_AttnQKV = calculate_alpha_given_nu_over_d_and_d(actual_nu_AttnQKV / d, d)

  estimate_alpha_XWQ = estimate_product_alpha(alpha_X, d, alpha_Q, d)
  estimate_alpha_XWK = estimate_product_alpha(alpha_X, d, alpha_K, d)
  estimate_alpha_QKT = estimate_product_alpha(estimate_alpha_XWQ, d, estimate_alpha_XWK, d)
  estimate_alpha_softmax_QKT = calculate_softmax_alpha(estimate_alpha_QKT)
  estimate_alpha_XWV = estimate_product_alpha(alpha_X, d, alpha_V, d)
  estimate_alpha_AttnQKV = estimate_product_alpha(estimate_alpha_softmax_QKT, d, estimate_alpha_XWV, d)
  estimate_nu_AttnQKV = d * calculate_nu_over_d_given_alpha_and_d(estimate_alpha_AttnQKV, d)

  if verbose:
    print("actual_alpha_AttnQKV = ", actual_alpha_AttnQKV, ", estimate_alpha_AttnQKV = ", estimate_alpha_AttnQKV)
    print("actual_nu_AttnQKV = ", actual_nu_AttnQKV, ", estimate_nu_AttnQKV = ", estimate_nu_AttnQKV)
    
  results_dict = {
    "actual_nu_AttnQKV" :  actual_nu_AttnQKV,
    "estimate_nu_AttnQKV" :  estimate_nu_AttnQKV,
    "actual_alpha_AttnQKV" :  actual_alpha_AttnQKV,
    "estimate_alpha_AttnQKV" : estimate_alpha_AttnQKV
  }

  return results_dict
  

def calculate_nu_Sigma_for_fixed_alpha_as_d_goes_to_infinity(
    alpha : float,
    num_iterations : int = 9
    ) -> tuple:
        
  # ** TO DO : adapt to GPU?

  d_vals = []
  nu_Sigma_vals = []
  nu_Sigma_over_d_vals = []
  nu_SigmaSquared_vals = []
  nu_SigmaSquared_over_d_vals = []
  initial_d = 1
  scale_factor = 10
  
  d = initial_d
  for i in range(num_iterations):
    nu_Sigma, nu_SigmaSquared = calculate_nu_Sigma_nu_SigmaSquared(d, alpha, uniform_draws = True)
    d_vals.append(d)
    nu_Sigma_vals.append(nu_Sigma)
    nu_Sigma_over_d_vals.append(nu_Sigma / d)
    nu_SigmaSquared_vals.append(nu_SigmaSquared)
    nu_SigmaSquared_over_d_vals.append(nu_SigmaSquared / d)
    print(alpha, d, nu_Sigma, nu_Sigma / d, nu_SigmaSquared, nu_SigmaSquared / d)
    d = scale_factor * d
    
    results_dict = {
        "d_vals" :  d_vals,
        "nu_Sigma_vals" : nu_Sigma_vals, 
        "nu_Sigma_over_d_vals" : nu_Sigma_over_d_vals, 
        "nu_SigmaSquared_vals" : nu_SigmaSquared_vals, 
        "nu_SigmaSquared_over_d_vals" : nu_SigmaSquared_over_d_vals
    }

    return results_dict
  
def calculate_nu_Sigma_nu_SigmaSquared(
                        d : int,
                        alpha : float,
                        uniform_draws : bool = False) -> tuple:
                            
  # uses correct alpha
  # ** TO DO : adapt to GPU?  
  
  these_sigmas = generate_pareto_draws(d, alpha, uniform_draws)
  nu_Sigma = calculate_nu(these_sigmas)

  these_sigmas_squared = these_sigmas ** 2
  nu_SigmaSquared = calculate_nu(these_sigmas_squared)

  return nu_Sigma, nu_SigmaSquared
  
def calculate_nu_W_nu_WTW(N : int,
                        d : int,
                        alpha : float,
                        uniform_draws : bool = False,
                        verbose : bool = False,
                        force_cpu : bool = False
                        ) -> dict:
                            
  # uses correct alpha
  # ** TO DO : adapt to GPU?  

  these_sigmas = generate_pareto_draws(d,alpha,uniform_draws)

  nu_Sigma = calculate_nu(these_sigmas)

  diag_sigmas = np.diag(these_sigmas)
  W0 = np.random.normal(0, 1, (N, d))
  QN = generate_orthogonal_matrix(N)
  Qd = generate_orthogonal_matrix(d)
  
  nu_W = 0.0 
  twonn_dim_W = 0.0
  nu_WTW = 0.0 
  twonn_dim_WTW = 0.0
  
  if use_gpu and not force_cpu:
    print(" ** calculate_nu_W_nu_WTW : using GPU **")
    cp_W1 = cp.matmul(cp.array(W0), cp.array(diag_sigmas))
    cp_W = cp.matmul(cp.matmul(cp.array(QN),cp_W1), cp.array(Qd))
    nu_W, twonn_dim_W = calculate_nu_twonn_dim(cp.asnumpy(cp_W))
    cp_WTW = cp.matmul(cp_W.T, cp_W)
    nu_WTW, twonn_dim_WTW = calculate_nu_twonn_dim(cp.asnumpy(cp_WTW))
  else:
    print(" ** calculate_nu_W_nu_WTW : using CPU **")
    W1 = W0 @ diag_sigmas
    W = QN @ W1 @ Qd
    nu_W, twonn_dim_W = calculate_nu_twonn_dim(W)
    WTW = W.T @ W
    nu_WTW, twonn_dim_WTW = calculate_nu_twonn_dim(WTW)

  results_dict = { 
    "nu_Sigma" : nu_Sigma,  
    "nu_W" : nu_W, 
    "nu_WTW" : nu_WTW,
    "twonn_dim_W" :  twonn_dim_W,
    "twonn_dim_WTW" : twonn_dim_WTW
    }
    
  return results_dict
  
def calculate_nu_W_nu_WTW_old(N : int,
                        d : int,
                        alpha : float,
                        uniform_draws : bool =False) -> dict:
                            
  # uses correct alpha
  # ** TO DO : adapt to GPU?   
  # ** TO DO : DEPRECATE?

  these_sigmas = np.zeros(d)
  for i in range(1,d+1): # 1 <= i <= d
    this_Fx = 0.0
    if uniform_draws:
        this_Fx = i / (d+1)
    else:
        this_Fx = np.random.uniform(0,1)
    this_x = (1.0/(1 - this_Fx))** (1.0/alpha)
    these_sigmas[i-1] = this_x

  nu_Sigma = calculate_nu(these_sigmas)

  diag_sigmas = np.diag(these_sigmas)
  W0 = np.random.normal(0, 1, (N, d))
  W1 = W0 @ diag_sigmas

  QN = generate_orthogonal_matrix(N)
  Qd = generate_orthogonal_matrix(d)

  W = QN @ W1 @ Qd

  nu_W, twonn_dim_W = calculate_nu_twonn_dim(W)

  WTW = W.T @ W
  nu_WTW, twonn_dim_WTW = calculate_nu_twonn_dim(WTW)

  results_dict = { 
    "nu_Sigma" : nu_Sigma,  
    "nu_W" : nu_W, 
    "nu_WTW" : nu_WTW,
    "twonn_dim_W" :  twonn_dim_W,
    "twonn_dim_WTW" : twonn_dim_WTW
    }
    
  return results_dict
  
def generate_pareto_draws(d : int, alpha : float, uniform_draws : bool = False) -> np.ndarray:
    
    """
    generate d draws from a Pareto distribution with tail exponent alpha
    uniform_draws = True : draw these at uniform cumulants of the CDF
    uniform_draws = False : draw these randomly from cumulants of the CDF
    """
    
    these_Fx = np.zeros(d)
    if uniform_draws:
        these_Fx = np.arange(1,d+1,1) / float(d+1.0)
        #this_Fx = i / (d+1)
    else:
        these_Fx = np.random.uniform(0,1,d)
        
    #print("these_Fx.shape = ", these_Fx.shape)
    return this_vec_inv_pareto_cdf(alpha, these_Fx)
  
def calculate_product_nu(d : int,
                         alpha_1 : float,
                         alpha_2 : float
                         ) -> dict:
                             
  # uses correct alpha
  # ** TO DO : adapt to GPU?

  these_sigmas_1 = np.zeros(d)
  for i in range(1,d+1): # 1 <= i <= d
    this_Fx = np.random.uniform(0,1)
    this_x = (1.0/(1 - this_Fx)) ** (1.0/alpha_1)
    these_sigmas_1[i-1] = this_x
  nu_Sigma1 = calculate_nu(these_sigmas_1)

  these_sigmas_2 = np.zeros(d)
  for i in range(1,d+1): # 1 <= i <= d
    this_Fx = np.random.uniform(0,1)
    this_x = (1.0/(1 - this_Fx)) ** (1.0/alpha_2)
    these_sigmas_2[i-1] = this_x
  nu_Sigma2 = calculate_nu(these_sigmas_2)

  Q11 = generate_orthogonal_matrix(d)
  Q12 = generate_orthogonal_matrix(d)
  Q21 = generate_orthogonal_matrix(d)
  Q22 = generate_orthogonal_matrix(d)
  
  W1 = Q11 @ np.diag(these_sigmas_1) @ Q12
  W2 = Q21 @ np.diag(these_sigmas_2) @ Q22
  nu_W1, twonn_dim_W1 = calculate_nu_twonn_dim(W1)
  nu_W2, twonn_dim_W2 = calculate_nu_twonn_dim(W2)

  W1W2 = W1 @ W2
  nu_W1W2, twonn_dim_W1W2 = calculate_nu_twonn_dim(W1W2)

  results_dict = {
    "nu_Sigma1" :  nu_Sigma1,
    "nu_Sigma2" : nu_Sigma2,
    "nu_W1" : nu_W1, 
    "nu_W2" : nu_W2, 
    "nu_W1W2" :  nu_W1W2,
    "twonn_dim_W1" :  twonn_dim_W1,
    "twonn_dim_W2" :  twonn_dim_W2,
    "twonn_dim_W1W2" : twonn_dim_W1W2
    }
    
  return results_dict
  
def generate_random_data_of_given_intrinsic_dim(embedding_dim : int,
                                                intrinsic_dim : int,
                                                num_embeddings : int,
                                                verbose : bool=False) -> np.ndarray:

  mu = 0.0
  sigma = 1.0

  random_square_1 = np.random.normal(mu, sigma, (embedding_dim, embedding_dim))
  random_square_2 = np.random.normal(mu, sigma, (num_embeddings, num_embeddings))

  random_intrinsic = np.random.normal(mu, sigma, (intrinsic_dim, num_embeddings))
  augmented_with_zeros = np.zeros((embedding_dim, num_embeddings))

  for row in range(random_intrinsic.shape[0]):
    augmented_with_zeros[row, :] = random_intrinsic[row, :]

  # rescale so that the entries of full_random have std_dev 1.0
  # using the formula
  # sigma (ABC) = sqrt(d_AB * d_BC) * sigma(A) * sigma(B) * sigma(C)
  # where d_AB is the common internal dimension of A and B
  # and d_BC is the common internal dimension of B and C
  full_random = np.matmul(np.matmul(random_square_1, augmented_with_zeros), random_square_2)
  scaling_factor = 1.0 / ((math.sqrt(intrinsic_dim * num_embeddings)) * (sigma * sigma * sigma))
  full_random = full_random * scaling_factor

  if verbose:
    print(random_square_1)
    print(random_square_2)
    print(random_intrinsic)
    print(augmented_with_zeros)
    print(full_random)

  return full_random
  
def generate_N_points_in_k_dimensional_subspace_in_R_d(
	N : int,
	d : int,
	k : int,
	manifold_type : str = "None",
	is_solid : bool = True,
	eccentricity : float = 1.0,
	displacement_scale : float = 0.0,
  no_orthogonal_multiplication : bool = False,
	verbose : bool = False,
  debug_mode : bool = False
	) -> np.ndarray:

	"""
	generate N points in R^d, with actual dimension k.
	N : number of points to generate
	d : dimension of full space
	k : desired dimension of data manifold
	"""
    # ** TO DO : adapt to GPU?

	is_cuboid = False
	is_ellipsoid = False

	if manifold_type == "cuboid":
		is_cuboid = True
	if manifold_type == "cube":
		is_cuboid = True
	if manifold_type == "ellipsoid":
		is_ellipsoid = True
	if manifold_type == "sphere":
		is_ellipsoid = True

	use_this_k = k
	if not is_solid:
		use_this_k = k + 1

	if debug_mode:
		print("k = ", k, " : use_this_k = ", use_this_k)

	X = np.zeros((N,d))
	eps = 1e-6

	if is_ellipsoid:
		# first, we generate N random vectors in R^d
		# second, we project onto a use_this_k -dimensional linear subspace
		# third, we normalise the vectors to unit length, which results in a (use_this_k - 1) -dimensional sphere
		# fourth, for solid only, we rescale so that the vectors have length between 0 and 1, with uniform sampling from the k-ball

		# one : generate
		X = np.random.randn(N, d)

		# two : project
		projection_matrix = np.zeros((d,d))
		for i in range(0,use_this_k):
			projection_matrix[i,i]=1.0
		X = X @ projection_matrix
		if debug_mode:
			print("projection_matrix = ", projection_matrix)

		# three : normalise
		for i in range(0,N):
			x_i = X[i,:]
			x_i = x_i / (eps + np.linalg.norm(x_i))
			X[i,:] = x_i

		if is_solid:
			#rescale
			t = np.random.uniform(0, 1, d)
			r = t ** (1.0/k)
			scale_matrix = np.diag(r)
			X = X @ scale_matrix

	if is_cuboid:
		# first, we generate N random vectors in R^d, drawing each entry from a uniform distribution between -1 and 1
		# second, we project onto a use_this_k -dimensional linear subspace
		# third, for non-solid, we randomly choose a coordinate between 0 and k, and set that coordinate to either -1 or 1 (with equal probability)

		# one : generate
		X = np.random.uniform(low = -1, high = +1, size = (N, d))

		# two : project
		projection_matrix = np.zeros((d,d))
		for i in range(0,use_this_k):
			projection_matrix[i,i]=1.0
		X = X @ projection_matrix

		# three : for non-solid, choose a coordinate index between 0 and k (inclusive),
		# and set the corresponding coordinate to -1 or +1
		if not is_solid:
			for i in range(0,N):
				this_index = np.random.randint(0,k+1) # 0 to k inclusive (but not k+1)
				X[i,this_index] = np.random.choice([-1,1])



	if verbose:
		plot_histogram_of_values(X,verbose)

	# multiply by eccentricities
	these_eccentricies = np.random.uniform(1, eccentricity, use_this_k)
	ecc_scale_matrix = np.zeros((d,d))
	for i in range(0,use_this_k):
		ecc_scale_matrix[i,i] = these_eccentricies[i]
	ecc_scale_matrix[0,0] = 1.0
	ecc_scale_matrix[use_this_k - 1, use_this_k - 1] = eccentricity
	X = X @ ecc_scale_matrix

	if verbose:
		plot_histogram_of_values(X,verbose)

	# multiply by orthogonal matrix
	if not no_orthogonal_multiplication:
		Q = generate_orthogonal_matrix(d)
		X = X @ Q

	if verbose:
		plot_histogram_of_values(X,verbose)

	#add displacement vector
	displacement_vector = np.zeros(d)
	if displacement_scale > 0.0:
		displacement_vector = np.random.randn(d)
		norm_displacement_vector = np.linalg.norm(displacement_vector)
		displacement_vector = displacement_vector / (eps + norm_displacement_vector)
		displacement_vector = displacement_vector * displacement_scale
	X = X + displacement_vector

	return X, displacement_vector
    
def run_experiment_one( N : int,
                        d : int,
                        k : int,
                        use_sphere : bool = False,
                        use_ball : bool = False,
                        use_surface_cube : bool = False,
                        use_solid_cube : bool = False,
                        use_surface_ellipsoid : bool = False,
                        #use_solid_ellipsoid : bool = False,
                        eccentricity : float = 1.0,
                        displacement_scale : float = 0.0,
                        verbose : bool = False
                        ) -> tuple[float, float]:

  # N : number of points in R^d to generate
  # d : the dimension of the ambient space R^d
  # k : the dimension of the subspace we're generating
  # use_sphere : whether to generate points on the k-sphere, or the k-ball
  # displacement_scale : displacement of the centre of the sphere / ball

  X = np.zeros((N,d))
  displacement_vector = np.zeros(d)

  if use_sphere:
    X, displacement_vector = generate_N_points_on_k_sphere_in_R_d(
                                          N,
                                          d,
                                          k,
                                          displacement_scale)
  if use_ball:
    X, displacement_vector = generate_N_points_in_unit_k_ball_in_R_d(
                                          N,
                                          d,
                                          k,
                                          displacement_scale)
  if use_surface_cube:
    X, displacement_vector = generate_N_points_on_k_dim_cubical_surface_in_R_d(
                                          N,
                                          d,
                                          k,
                                          displacement_scale)

  if use_solid_cube:
    X, displacement_vector = generate_N_points_in_solid_k_cube_in_R_d(
                                          N,
                                          d,
                                          k,
                                          displacement_scale)
  if use_surface_ellipsoid:
    X, displacement_vector = generate_N_points_on_surface_of_k_ellipsoid_in_R_d(
                                          N,
                                          d,
                                          k,
                                          eccentricity,
                                          displacement_scale)

  if verbose:
    plot_histogram_of_values(X, verbose=True, rescale_factor=1.0, num_bins=100)

  # calculate mu, the average of the N column vectors
  mu = X.sum(axis=0) / N
  if verbose:
    plot_histogram_of_values(mu, verbose=True, rescale_factor=1.0, num_bins=100)

  Xdemeaned = X - mu
  if verbose:
    plot_histogram_of_values(Xdemeaned, verbose=True, rescale_factor=1.0, num_bins=100)

  U, S, Vh = np.linalg.svd(Xdemeaned, full_matrices=True)

  # Print the shapes of the resulting matrices
  if verbose:
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)

  # Reconstruct the original matrix
  Sigma = np.zeros((this_N, this_d))
  for i in range(this_d):
    Sigma[i, i] = S[i]
  Xdemeaned_reconstructed = U @ Sigma @ Vh

  # Verify the reconstruction
  if verbose:
    print("Reconstruction error:", np.linalg.norm(Xdemeaned - Xdemeaned_reconstructed))

  if verbose:
    plot_histogram_of_values(S)

  nu = calculate_nu(S)
  if verbose:
    print("nu = ", nu)

  twonn_dim = twonn_intrinsic_dimension(X, plot_fit=verbose)

  return nu, twonn_dim
  
def generate_N_points_on_k_sphere_in_R_d(N : int,
                                         d : int,
                                         k : int,
                                         displacement_scale : float = 0.0
                                         ) -> np.ndarray:
  """
  generate N points in R^d, with actual dimension k.
  N : number of points to generate
  d : dimension of full space
  k : actual dimension
  first, we generate N random vectors in R^d
  second, we project onto a (k+1)-dimensional linear subspace
  third, we normalise the vectors to unit length, which results in a k-dimensional sphere
  fourth, we multiply by an orthogonal matrix, so the vectors are no longer restricted to the subspace x_{k+2} = .. = x_d = 0
  fifth, we add a displacement vector, to shift the points away from the origin
  sixth, we return the points and the displacement vector
  """
  
  # ** TO DO : adapt to GPU?
  
  X = np.random.randn(N, d)
  Pkplusone = np.zeros((d,d))
  for i in range(0,k+1):
    Pkplusone[i,i] = 1
  X = X @ Pkplusone

  #normalise
  eps = 1e-6
  for i in range(0,N):
    xi = X[i,:]
    xi = xi / (eps + np.linalg.norm(xi))
    X[i,:] = xi / (eps + np.linalg.norm(xi))

  # multiply by orthogonal matrix
  Q = generate_orthogonal_matrix(d)
  X = X @ Q

  #add displacement vector
  displacement_vector = np.zeros(d)
  if displacement_scale > 0.0:
    displacement_vector = np.random.randn(d)
    norm_displacement_vector = np.linalg.norm(displacement_vector)
    displacement_vector = displacement_vector / (eps + norm_displacement_vector)
    displacement_vector = displacement_vector * displacement_scale
  X = X + displacement_vector

  return X, displacement_vector

def generate_N_points_in_unit_k_ball_in_R_d(N : int,
                                                d : int,
                                                k : int,
                                                displacement_scale : float = 0.0
                                                ) -> np.ndarray:
  """
  generate N points in R^d, with actual dimension k.
  N : number of points to generate
  d : dimension of full space
  k : actual dimension
  first, we generate N random vectors in R^d
  second, we project onto a k-dimensional linear subspace
  third, we normalise the vectors to unit length, which results in a (k-1)-dimensional sphere - the surface of the k-ball
  fourth, we rescale so that the vectors have length between 0 and 1, with uniform sampling from the k-ball
  fourth, we multiply by an orthogonal matrix, so the vectors are no longer restricted to the subspace x_{k+1} = .. = x_d = 0
  fifth, we add a displacement vector, to shift the points away from the origin
  sixth, we return the points and the displacement vector
  """
  
  # ** TO DO : adapt to GPU?
  
  X = np.random.randn(N, d)
  Pk = np.zeros((d,d))
  for i in range(0,k):
    Pk[i,i] = 1
  X = X @ Pk

  #normalise
  eps = 1e-6
  for i in range(0,N):
    xi = X[i,:]
    xi = xi / (eps + np.linalg.norm(xi))
    X[i,:] = xi / (eps + np.linalg.norm(xi))

  #rescale
  t = np.random.uniform(0, 1, N)
  r = t ** (1.0/k)
  scale_matrix = np.diag(r)
  X = scale_matrix @ X

  # multiply by orthogonal matrix
  Q = generate_orthogonal_matrix(d)
  X = X @ Q

  #add displacement vector
  displacement_vector = np.zeros(d)
  if displacement_scale > 0.0:
    displacement_vector = np.random.randn(d)
    norm_displacement_vector = np.linalg.norm(displacement_vector)
    displacement_vector = displacement_vector / (eps + norm_displacement_vector)
    displacement_vector = displacement_vector * displacement_scale
  X = X + displacement_vector

  return X, displacement_vector
  
def generate_N_points_on_surface_of_k_ellipsoid_in_R_d(N : int,
                                         d : int,
                                         k : int,
                                         displacement_scale : float = 0.0,
                                         eccentricity : float = 1.0,
                                         ) -> np.ndarray:
  """
  generate N points in R^d, with actual dimension k.
  N : number of points to generate
  d : dimension of full space
  k : actual dimension
  first, we generate N random vectors in R^d
  second, we project onto a (k+1)-dimensional linear subspace
  third, we normalise the vectors to unit length, which results in a k-dimensional sphere
  third + one, we generate (k+1) eccentricities, each in the range [1, eccentricity], and scale each dimension accordingly
  fourth, we multiply by an orthogonal matrix, so the vectors are no longer restricted to the subspace x_{k+2} = .. = x_d = 0
  fifth, we add a displacement vector, to shift the points away from the origin
  sixth, we return the points and the displacement vector
  """
  
  # ** TO DO : adapt to GPU?
  
  X = np.random.randn(N, d)
  Pkplusone = np.zeros((d,d))
  for i in range(0,k+1):
    Pkplusone[i,i] = 1
  X = X @ Pkplusone

  #normalise
  eps = 1e-6
  for i in range(0,N):
    xi = X[i,:]
    xi = xi / (eps + np.linalg.norm(xi))
    X[i,:] = xi / (eps + np.linalg.norm(xi))

  # generate eccentricities
  these_eccentricies = np.random.uniform(1, eccentricity, k+1)
  ecc_scale_matrix = np.zeros((d,d))
  for i in range(0,k+1):
    ecc_scale_matrix[i,i] = these_eccentricies[i]
  X = X @ ecc_scale_matrix

  # multiply by orthogonal matrix
  Q = generate_orthogonal_matrix(d)
  X = X @ Q

  #add displacement vector
  displacement_vector = np.zeros(d)
  if displacement_scale > 0.0:
    displacement_vector = np.random.randn(d)
    norm_displacement_vector = np.linalg.norm(displacement_vector)
    displacement_vector = displacement_vector / (eps + norm_displacement_vector)
    displacement_vector = displacement_vector * displacement_scale
  X = X + displacement_vector

  return X, displacement_vector
  
def generate_N_points_in_solid_k_cube_in_R_d(N : int,
                                          d : int,
                                          k : int,
                                          displacement_scale : float = 0.0,
                                          verbose : bool = False,
                                          no_orthogonal_multiplication : bool = False,
                                          eccentricity : float = 1.0
                                          ) -> tuple[np.ndarray, np.ndarray]:
                                              
    # ** TO DO : adapt to GPU?

    this_solid_cube = True
    kminusone = k-1

    X, displacement_vector = generate_N_points_on_k_dim_cubical_surface_in_R_d(
                                          N,
                                          d,
                                          kminusone,
                                          displacement_scale,
                                          verbose,
                                          this_solid_cube,
                                          no_orthogonal_multiplication,
                                          eccentricity
                                          )

    return X, displacement_vector
    
def generate_N_points_on_k_dim_cubical_surface_in_R_d(
                                          N : int,
                                          d : int,
                                          k : int,
                                          displacement_scale : float = 0.0,
                                          verbose : bool = False,
                                          solid_cube : bool = False,
                                          no_orthogonal_multiplication : bool = False,
                                          eccentricity : float = 1.0
                                          ) -> tuple[np.ndarray, np.ndarray]:
  """
  generate N points in R^d, on a k-dimensional cubical surface
  so the expected dimension of this space is k
  N : number of points to generate
  d : dimension of full space
  k : dimension of the cubical surface - the enclosed solid cube will have dimension k+1
  first, we generate N random vectors in R^d, drawing each entry from a uniform distribution between -1 and 1
  second, we project onto a (k+1)-dimensional linear subspace
  third, we randomly choose a coordinate between 0 and k, and set that coordinate to either -1 or 1 (with equal probability)
  third + onely, we generate (k+1) eccentricities, each in the range [1, eccentricity], and scale each dimension accordingly
  fourth, we multiply by a d*d orthogonal matrix, so the vectors are no longer restricted to the subspace x_{k+2} = .. = x_d = 0
  fifth, we add a displacement vector, to shift the points away from the origin
  sixth, we return the points and the displacement vector
  """
  
  # ** TO DO : adapt to GPU?
  
  # one : choose N random points
  X = np.random.uniform(low = -1, high = +1, size = (N, d))

  # two : project onto a (k+1)-dimensional subspace
  Pkplusone = np.zeros((d,d))
  for i in range(0,min(d,k+1)):
    Pkplusone[i,i] = 1
  X = X @ Pkplusone

  # three : choose a coordinate index between 0 and k (inclusive),
  # and set the corresponding coordinate to -1 or +1
  if not solid_cube:
    for i in range(0,N):
      this_index = np.random.randint(0,k+1) # 0 to k inclusive (but not k+1)
      X[i,this_index] = np.random.choice([-1,1])
  if verbose:
    plot_histogram_of_values(X,verbose)

  # three.one : generate eccentricities
  # enforce that the eccentricities 1.0 and eccentricity both appear
  ecc_scale_matrix = np.zeros((d,d))
  ecc_scale_matrix[0,0] = 1.0
  for i in range(1,k):
    ecc_scale_matrix[i,i] = np.random.uniform(1, eccentricity)
  ecc_scale_matrix[k,k] = eccentricity
  X = X @ ecc_scale_matrix

  # four : multiply by orthogonal matrix
  if not no_orthogonal_multiplication:
    Q = generate_orthogonal_matrix(d)
    X = X @ Q

  # five : add displacement vector
  eps = 1e-6
  displacement_vector = np.zeros(d)
  if displacement_scale > 0.0:
    displacement_vector = np.random.randn(d)
    norm_displacement_vector = np.linalg.norm(displacement_vector)
    displacement_vector = displacement_vector / (eps + norm_displacement_vector)
    displacement_vector = displacement_vector * displacement_scale
  X = X + displacement_vector

  return X, displacement_vector
  
def calculate_alpha_given_nu_over_d_and_d(nu_over_d : float,
                                          d : int,
                                          verbose : bool = False
                                          ) -> float:
  # given nu / d and d, find alpha
  # uses new definition of alpha 

  eps = 1e-9

  # initial estimates
  alpha_0 = 5.0 * nu_over_d # NEW 02/06
  #alpha_0 = 1.0 + (1.0 - nu_over_d)**0.5 # NEW
  #alpha_0 = 2.0 + (1.0 - nu_over_d)**0.5 # OLD
  _, nu_over_d_0 = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha_0, d)

  alpha_i = alpha_0
  nu_over_d_i = nu_over_d_0
  alpha_i_plus_one = alpha_i
  nu_over_d_i_plus_one = nu_over_d_i

  stepwise_scale_factor = 0.99
  delta_alpha = 0.1
  min_alpha_i = 0.01 # was 1.01
  for i in range(0,100):
    if abs(nu_over_d_i - nu_over_d) < eps:
      return alpha_i
    # newton-raphson method
    new_nu, new_nu_over_d = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha_i, d)
    g_i = new_nu_over_d - nu_over_d
    _, new_nu_over_d_plus = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha_i + delta_alpha, d)
    _, new_nu_over_d_minus = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha_i - delta_alpha, d)
    g_prime_i = (new_nu_over_d_plus - new_nu_over_d_minus) / (2.0 * delta_alpha)

    if abs(g_prime_i) < eps:
      delta_alpha *=2
      alpha_i_plus_one = alpha_i
    else:
      alpha_i_plus_one = alpha_i - (g_i / g_prime_i)

    if alpha_i_plus_one < min_alpha_i:
      alpha_i_plus_one = 0.5 * (min_alpha_i + alpha_i)
      
    _, nu_over_d_i_plus_one = calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha_i_plus_one,d)

    alpha_i = alpha_i_plus_one
    nu_over_d_i = nu_over_d_i_plus_one
    delta_alpha = delta_alpha * stepwise_scale_factor

    if verbose:
      print(i, "delta_alpha = ", delta_alpha, ", alpha_i = ", alpha_i, ", nu_over_d_i = ", nu_over_d_i)

  return alpha_i

  
def calculate_alpha_given_nu_over_d_and_d_old(nu_over_d : float,
                                          d : int,
                                          verbose : bool = False
                                          ) -> float:
                                              
  # ** TO DO : deprecate                                            
                                               
  # old and spare - 02/05/2026 : I compared old and new versions, get identical results
  # given nu / d and d, find alpha

  eps = 1e-9

  # initial estimates
  alpha_0 = 2 + (1.0 - nu_over_d)**0.5
  _, nu_over_d_0 = calculate_nu_alpha_d_analytic(alpha_0, d)

  alpha_i = alpha_0
  nu_over_d_i = nu_over_d_0
  alpha_i_plus_one = alpha_i
  nu_over_d_i_plus_one = nu_over_d_i

  stepwise_scale_factor = 0.99
  delta_alpha = 0.1
  min_alpha_i = 1.01
  for i in range(0,100):
    if abs(nu_over_d_i - nu_over_d) < eps:
      return alpha_i
    # newton-raphson method
    new_nu, new_nu_over_d = calculate_nu_alpha_d_analytic(alpha_i, d)
    g_i = new_nu_over_d - nu_over_d
    _, new_nu_over_d_plus = calculate_nu_alpha_d_analytic(alpha_i + delta_alpha, d)
    _, new_nu_over_d_minus = calculate_nu_alpha_d_analytic(alpha_i - delta_alpha, d)
    g_prime_i = (new_nu_over_d_plus - new_nu_over_d_minus) / (2.0 * delta_alpha)

    if abs(g_prime_i) < eps:
      delta_alpha *=2
      alpha_i_plus_one = alpha_i
    else:
      alpha_i_plus_one = alpha_i - (g_i / g_prime_i)
      
    if alpha_i_plus_one < min_alpha_i:
      alpha_i_plus_one = 0.5 * (min_alpha_i + alpha_i)

    _, nu_over_d_i_plus_one = calculate_nu_alpha_d_analytic(alpha_i_plus_one,d)
    
    alpha_i = alpha_i_plus_one
    nu_over_d_i = nu_over_d_i_plus_one
    delta_alpha = delta_alpha * stepwise_scale_factor

    if verbose:
      print(i, delta_alpha, alpha_i, nu_over_d_i)

  return alpha_i
  
def calculate_nu_and_nu_over_d_given_alpha_d_analytic_array(alpha_vals : np.ndarray, d : int) -> tuple:

  return calculate_nu_alpha_d_analytic_array(alpha_vals, d, warning=False)
  
def calculate_nu_alpha_d_analytic_array(alpha_vals : np.ndarray, d : int, warning=True) -> tuple:
    
  if warning:
    print("** warning: call calculate_nu_and_nu_over_d_given_alpha_d_analytic_array rather than calculate_nu_alpha_d_analytic_array **")
    
  nu_over_d_vals = []
  for alpha in alpha_vals:
    nu, nu_over_d = calculate_nu_alpha_d_analytic(alpha, d)
    nu_over_d_vals.append(nu_over_d)
  return np.array(nu_over_d_vals)
  
def calculate_nu_and_nu_over_d_given_alpha_d_analytic(alpha : float, d : int) -> tuple:

  return calculate_nu_alpha_d_analytic(alpha, d, warning=False)

def calculate_nu_alpha_d_analytic(alpha : float, d : int, warning=True) -> tuple:
  # uses correct alpha 
  # ** TO DO : deprecate this

  if warning:
    print("** warning: call calculate_nu_and_nu_over_d_given_alpha_d_analytic rather than calculate_nu_alpha_d_analytic **")

  eps = 1e-6
  eps = 1e-6

  default_nu = 0.0
  default_nu_over_d = 0.0

  #s = 1.0 / (alpha - 1.0)
  s = 1.0 / alpha

  if alpha < 0.0:
    return default_nu, default_nu_over_d

  if abs(alpha - 2.0) < eps:
    nu_over_d = 4.0 / math.log(d)
    nu = nu_over_d * d
    return nu, nu_over_d

  if abs(alpha - 1.0) < eps:
    nu = (math.log(d))**2
    nu_over_d = nu / d
    return nu, nu_over_d

  if alpha > 2.0:
    nu_over_d = C_alpha(alpha) * ((1.0 - d**(s - 1.0))**2) / (1.0 - d**(2.0*s - 1.0))
    nu = nu_over_d * d
    return nu, nu_over_d

  if 2.0 > alpha and alpha > 1.0:
    nu_over_d = -1.0 * C_alpha(alpha) * (d**(1.0 - 2.0 * s)) * ((1 - d**(s - 1.0))**2) / (1.0 - d**(1.0 - 2.0 * s))
    nu = nu_over_d * d
    return nu, nu_over_d

  if 1.0 > alpha:
    nu = -1.0 * C_alpha(alpha) * ((1.0 - d**(1.0 - s))**2) / (1.0 - d**(1.0 - 2.0*s))
    nu_over_d = nu / d
    return nu, nu_over_d

  return default_nu, default_nu_over_d
  
def get_nu_over_d_vals_from_alpha_vals_d(alpha_vals : np.ndarray, d : int) -> np.ndarray:
  nu_over_d_vals = []
  for alpha in alpha_vals:
    nu, nu_over_d = calculate_nu_alpha_d_analytic(alpha, d)
    nu_over_d_vals.append(nu_over_d)
    return np.array(nu_over_d_vals)
  
def calculate_conjectured_limiting_value_nu_over_d(alpha : float) -> float:
  # conjectured limiting value as d -> infinity
  # uses correct alpha 
  if alpha <= 2.0:
    return 0.0
  else:
    return C_alpha(alpha)

def C_alpha(alpha : float) -> float:
  # uses correct alpha  
  return ((alpha - 2.0) * alpha) / (alpha - 1.0)**2
  
def C_alpha_old(alpha : float) -> float:
  # ** TO DO : OLD DEFINITION
  return ((alpha - 3.0) * (alpha - 1.0)) / (alpha - 2.0)**2
  
def smooth_this_array(input_array : np.ndarray,
                      smoothing_window_length : int = 10,
                      num_decimal_places : int = 3) -> np.ndarray:

  len_input_array = input_array.shape[0]
  smoothing_parameters = np.ones(smoothing_window_length)
  for i in range(smoothing_window_length):
    smoothing_parameters[i] = min(i + 1, smoothing_window_length - i)
  print(smoothing_parameters)
  smoothing_parameters = smoothing_parameters / np.sum(smoothing_parameters)
  print(smoothing_parameters)

  len_input_array = input_array.shape[0]
  len_extended_array = len_input_array + 2 * (smoothing_window_length - 1)

  extended_input_array = np.zeros(len_extended_array)
  for i in range(len_extended_array):
    this_index = max(0, min(i - smoothing_window_length + 1, len_input_array - 1))
    extended_input_array[i] = input_array[this_index]

  smoothed_values_initial = np.convolve(extended_input_array, smoothing_parameters, mode='same')
  smoothed_values_final = smoothed_values_initial[smoothing_window_length - 2 : -smoothing_window_length]

  dp_factor = 10**num_decimal_places
  for i in range(len(smoothed_values_final)):
    smoothed_values_final[i] = round(smoothed_values_final[i] * dp_factor) / dp_factor

  return smoothed_values_final
  
# apply log to an array by vectorisation
def this_abs(x):
  return abs(x)

# Apply the function using np.vectorize
this_vec_abs = np.vectorize(this_abs)

# apply log to an array by vectorisation
def this_log10(x):
  eps=1e-6
  return math.log10(max(x,eps))

# Apply the function using np.vectorize
this_vec_log10 = np.vectorize(this_log10)

def product_alpha_experiment(N : int,
                             d : int,
                             alpha_X : float,
                             alpha_W : float
                             ) -> dict:
                                 
  # ** TO DO : adapt to GPU?

  this_N = N
  this_d = d

  X = generate_data_manifold(this_N, this_d, alpha_X)
  W = generate_square_weight_matrix(this_d, alpha_W)

  pp_dim_X = calculate_PatnaikPearson_dim(X, verbose=False)
  pp_dim_W = calculate_PatnaikPearson_dim(W, verbose=False)

  dim_X = X.shape[1]
  dim_W = W.shape[1]

  nu_over_d_X = pp_dim_X / dim_X
  nu_over_d_W = pp_dim_W / dim_W

  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X, dim_X)
  actual_alpha_W = calculate_alpha_given_nu_over_d_and_d(nu_over_d_W, dim_W)

  XW = X @ W
  pp_dim_XW = calculate_PatnaikPearson_dim(XW, verbose=False)
  nu_over_d_XW = pp_dim_XW / this_d
  alpha_XW = calculate_alpha_given_nu_over_d_and_d(nu_over_d_XW, this_d)

  nu_over_d_XW_estimate = estimate_product_nu_over_d(nu_over_d_X, this_d, nu_over_d_W, this_d)
  pp_dim_XW_estimate = nu_over_d_XW_estimate * this_d
  alpha_XW_estimate = calculate_alpha_given_nu_over_d_and_d(nu_over_d_XW_estimate, this_d)

  results_dict = {}
  results_dict["alpha_XW"] = alpha_XW
  results_dict["alpha_XW_estimate"] = alpha_XW_estimate
  results_dict["actual_alpha_X"] = actual_alpha_X
  results_dict["actual_alpha_W"] = actual_alpha_W
  results_dict["pp_dim_X"] = pp_dim_X
  results_dict["pp_dim_W"] = pp_dim_W
  results_dict["pp_dim_XW"] = pp_dim_XW
  results_dict["nu_over_d_X"] = nu_over_d_X
  results_dict["nu_over_d_W"] = nu_over_d_W
  results_dict["nu_over_d_XW"] = nu_over_d_XW
  results_dict["nu_over_d_XW_estimate"] = nu_over_d_XW_estimate

  return results_dict
  

def relu_experiment(N : int,
                    d : int,
                    alpha_X : float,
                    generate_weight_matrix : bool = False,
                    uniform_draws : bool = False,
                    use_pareto : bool = True,
                    use_uniform : bool = False,
                    use_cauchy : bool = False,
                    verbose : bool = False
                    ) -> dict:
                        
                        

  #X = generate_data_manifold(N, d, alpha_X)
  
  X = np.zeros((N,d))
  if generate_weight_matrix:
      X = generate_square_weight_matrix(d, 
                                  alpha_X,
                                  uniform_draws,
                                  use_pareto,
                                  use_uniform,
                                  use_cauchy,
                                  verbose)
  else:
      X = generate_data_manifold(N, 
                    d, 
                    alpha_X,
                    uniform_draws,
                    use_pareto,
                    use_uniform,
                    use_cauchy,
                    verbose)
  
  dim_X = X.shape[1]
  pp_dim_X = calculate_PatnaikPearson_dim(X)
  nu_over_d_X = pp_dim_X / dim_X
  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X, dim_X)

  if verbose:
    print("dim_X = ", dim_X)
    print("pp_dim_X = ", pp_dim_X)
    print("nu_over_d_X = ", nu_over_d_X)
    print("actual_alpha_X = ", actual_alpha_X)

  reluX = this_vec_relu(X)
  #print(reluX[0:n,0:n])
  dim_reluX = reluX.shape[1]
  pp_dim_reluX = calculate_PatnaikPearson_dim(reluX)
  nu_over_d_reluX = pp_dim_reluX / dim_reluX
  actual_alpha_reluX = calculate_alpha_given_nu_over_d_and_d(nu_over_d_reluX, dim_reluX)

  if verbose:
    print("dim_reluX = ", dim_reluX)
    print("pp_dim_reluX = ", pp_dim_reluX)
    print("nu_over_d_reluX = ", nu_over_d_reluX)
    print("actual_alpha_reluX = ", actual_alpha_reluX)

  results_dict = {
	"actual_alpha_X" : actual_alpha_X,
	"actual_alpha_reluX" : actual_alpha_reluX,
	"pp_dim_X" : pp_dim_X, 
	"pp_dim_reluX" : pp_dim_reluX,
	"nu_over_d_X" : nu_over_d_X, 
	"nu_over_d_reluX" : nu_over_d_reluX
	}

  return results_dict
 

def sigmoid_experiment(N : int,
                    d : int,
                    alpha_X : float,
                    generate_weight_matrix : bool = False,
                    pareto_uniform_draws : bool = False,
                    use_pareto : bool = False,
                    use_uniform : bool = False,
                    use_cauchy : bool = False,
                    verbose : bool = False
                    ) -> dict:
                        
  X = np.zeros((N,d))
  
  if generate_weight_matrix:
      X = generate_square_weight_matrix(d, 
                                  alpha_X,
                                  pareto_uniform_draws,
                                  use_pareto,
                                  use_uniform,
                                  use_cauchy,
                                  verbose
                                  )
  else:
      X = generate_data_manifold(N, 
                    d, 
                    alpha_X, 
                    pareto_uniform_draws,
                    use_pareto,
                    use_uniform,
                    use_cauchy)
                    				
  dim_X = X.shape[1]
  pp_dim_X = calculate_PatnaikPearson_dim(X)
  nu_over_d_X = pp_dim_X / dim_X
  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X, dim_X)

  if verbose:
    print("dim_X = ", dim_X)
    print("pp_dim_X = ", pp_dim_X)
    print("nu_over_d_X = ", nu_over_d_X)
    print("actual_alpha_X = ", actual_alpha_X)

  sigmoidX = this_vec_sigmoid(X)
  #print(sigmoidX[0:n,0:n])
  dim_sigmoidX = sigmoidX.shape[1]
  pp_dim_sigmoidX = calculate_PatnaikPearson_dim(sigmoidX)
  nu_over_d_sigmoidX = pp_dim_sigmoidX / dim_sigmoidX
  actual_alpha_sigmoidX = calculate_alpha_given_nu_over_d_and_d(nu_over_d_sigmoidX, dim_sigmoidX)

  if verbose:
    print("dim_sigmoidX = ", dim_sigmoidX)
    print("pp_dim_sigmoidX = ", pp_dim_sigmoidX)
    print("nu_over_d_sigmoidX = ", nu_over_d_sigmoidX)
    print("actual_alpha_sigmoidX = ", actual_alpha_sigmoidX)

  results_dict = {
	"actual_alpha_X" : actual_alpha_X,
	"actual_alpha_sigmoidX" : actual_alpha_sigmoidX,
	"pp_dim_X" : pp_dim_X, 
	"pp_dim_sigmoidX" : pp_dim_sigmoidX,
	"nu_over_d_X" : nu_over_d_X, 
	"nu_over_d_sigmoidX" : nu_over_d_sigmoidX
	}

  return results_dict
  

 
def addition_experiment(N : int,
                        d : int,
                        alpha_X1 : float,
                        alpha_X2 : float
                        ) -> dict:
                            
  """
  generate data manifolds X1, X2, both of shape (N,d) 
  with tail exponents alpha_X1, alpha_X2
  add them together (elementwise) : X1 + X2
  calculate Patnaik-Pearson dimension, nu/d and implied alpha for X1, X2 and X1 + X2
  """

  X1 = generate_data_manifold(N, d, alpha_X1)
  dim_X1 = X1.shape[1]
  pp_dim_X1 = calculate_PatnaikPearson_dim(X1)
  nu_over_d_X1 = pp_dim_X1 / dim_X1
  actual_alpha_X1 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1, dim_X1)

  X2 = generate_data_manifold(N, d, alpha_X2)
  dim_X2 = X2.shape[1]
  pp_dim_X2 = calculate_PatnaikPearson_dim(X2)
  nu_over_d_X2 = pp_dim_X2 / dim_X2
  actual_alpha_X2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X2, dim_X2)

  X1plusX2 = X1 + X2
  dim_X1plusX2 = X1plusX2.shape[1]
  pp_dim_X1plusX2 = calculate_PatnaikPearson_dim(X1plusX2)
  nu_over_d_X1plusX2 = pp_dim_X1plusX2 / dim_X1plusX2
  actual_alpha_X1plusX2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1plusX2, dim_X1plusX2)

  results_dict = { 
    "actual_alpha_X1" : actual_alpha_X1, 
    "actual_alpha_X2" : actual_alpha_X2, 
    "actual_alpha_X1plusX2" : actual_alpha_X1plusX2, 
    "pp_dim_X1" : pp_dim_X1,
    "pp_dim_X2" : pp_dim_X2,
    "pp_dim_X1plusX2" : pp_dim_X1plusX2,
    "nu_over_d_X1" : nu_over_d_X1,
    "nu_over_d_X2" : nu_over_d_X2, 
    "nu_over_d_X1plusX2" : nu_over_d_X1plusX2
  }
  
  return results_dict
  
def addition_experiment_two(N : int,
                        d : int,
                        alpha_X1 : float,
                        alpha_X2 : float,
                        pareto_uniform_draws : bool = False,
                        force_pareto : bool = False,
                        force_uniform : bool = False,
                        force_cauchy : bool = False,
                        verbose : bool = False
                        ) -> dict:
                            
  """
  generate data manifolds X1, X2, both of shape (N,d) 
  using either Uniform, Pareto or Cauchy distributions
  with tail exponents alpha_X1, alpha_X2
  add them together (elementwise) : X1 + X2
  calculate Patnaik-Pearson dimension, nu/d and implied alpha for X1, X2 and X1 + X2
  """
  
  if verbose:
      print("force_pareto = ", force_pareto)
      print("force_uniform = ", force_uniform)
      print("force_cauchy = ", force_cauchy)
      
  uniform_draws = False

  this_random = np.random.randint(0, 3)
  use_pareto = this_random == 0
  use_uniform = this_random == 1
  use_cauchy = this_random == 2
  this_alpha = np.random.uniform(0.1,5.0)

  if force_pareto:
        use_pareto = True
        uniform_draws = pareto_uniform_draws 
        this_alpha = alpha_X1
        use_uniform = False
        use_cauchy = False
  if force_uniform:
        use_pareto = False
        use_uniform = True
        use_cauchy = False
  if force_cauchy:
        use_pareto = False
        use_uniform = False
        use_cauchy = True
        
  if verbose:
        print("X1 : use_pareto = ", use_pareto)
        print("X1 : use_uniform = ", use_uniform)
        print("X1 : use_cauchy = ", use_cauchy)
        
  X1 = generate_data_manifold(N, 
                              d, 
                              this_alpha, 
                              uniform_draws,
                              use_pareto,
                              use_uniform,
                              use_cauchy)
  dim_X1 = X1.shape[1]
  pp_dim_X1 = calculate_PatnaikPearson_dim(X1)
  nu_over_d_X1 = pp_dim_X1 / dim_X1
  actual_alpha_X1 = 0.0
  if force_pareto:
    actual_alpha_X1 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1, dim_X1)

  this_random = np.random.randint(0, 3)
  use_pareto = this_random == 0
  use_uniform = this_random == 1
  use_cauchy = this_random == 2
  this_alpha = np.random.uniform(0.1,5.0)

  if force_pareto:
        use_pareto = True
        uniform_draws = pareto_uniform_draws 
        this_alpha = alpha_X2
        use_uniform = False
        use_cauchy = False
  if force_uniform:
        use_pareto = False
        use_uniform = True
        use_cauchy = False
  if force_cauchy:
        use_pareto = False
        use_uniform = False
        use_cauchy = True
        
  if verbose:
        print("X2 : use_pareto = ", use_pareto)
        print("X2 : use_uniform = ", use_uniform)
        print("X2 : use_cauchy = ", use_cauchy)
        
  X2 = generate_data_manifold(N, 
                              d, 
                              this_alpha, 
                              uniform_draws,
                              use_pareto,
                              use_uniform,
                              use_cauchy)
  dim_X2 = X2.shape[1]
  pp_dim_X2 = calculate_PatnaikPearson_dim(X2)
  nu_over_d_X2 = pp_dim_X2 / dim_X2
  actual_alpha_X2 = 0.0
  if force_pareto:
    actual_alpha_X2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X2, dim_X2)

  X1plusX2 = X1 + X2
  dim_X1plusX2 = X1plusX2.shape[1]
  pp_dim_X1plusX2 = calculate_PatnaikPearson_dim(X1plusX2)
  nu_over_d_X1plusX2 = pp_dim_X1plusX2 / dim_X1plusX2
  actual_alpha_X1plusX2 = 0.0
  if force_pareto:
    actual_alpha_X1plusX2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1plusX2, dim_X1plusX2)

  results_dict = { 
    "actual_alpha_X1" : actual_alpha_X1, 
    "actual_alpha_X2" : actual_alpha_X2, 
    "actual_alpha_X1plusX2" : actual_alpha_X1plusX2, 
    "pp_dim_X1" : pp_dim_X1,
    "pp_dim_X2" : pp_dim_X2,
    "pp_dim_X1plusX2" : pp_dim_X1plusX2,
    "nu_over_d_X1" : nu_over_d_X1,
    "nu_over_d_X2" : nu_over_d_X2, 
    "nu_over_d_X1plusX2" : nu_over_d_X1plusX2
  }
  
  return results_dict

def concatenation_experiment(N : int,
                             d1 : int,
                             d2 : int,
                             alpha_X1 : float,
                             alpha_X2 : float,
                             pareto_uniform_draws : bool = False,
                             force_pareto : bool = False,
                             force_uniform : bool = False,
                             force_cauchy : bool = False,
                             verbose : bool = False
                            ) -> dict:
                            
  """
  generate data manifolds X1, X2, of shapes (N,d1) and (N,d2) 
  using either Uniform, Pareto or Cauchy distributions
  with tail exponents alpha_X1, alpha_X2
  concatenate them : X12 = X1 oplus X2, to give shape (N, d1 + d2)
  calculate Patnaik-Pearson dimension, nu/d and implied alpha for X1, X2 and X12
  """
  
  if verbose:
      print("force_pareto = ", type(force_pareto), force_pareto)
      print("force_uniform = ", type(force_uniform), force_uniform)
      print("force_cauchy = ", type(force_cauchy), force_cauchy)
      
  uniform_draws = False

  #this_random = np.random.randint(0, 3)
  this_random = int(np.random.uniform(0,3))
  print("this_random = ", this_random, this_random == 0, this_random == 1, this_random == 2)
  use_pareto = this_random == 0
  use_uniform = this_random == 1
  use_cauchy = this_random == 2
  print(use_pareto, use_uniform, use_cauchy)
  this_alpha = np.random.uniform(0.1,5.0)

  if force_pareto:
        use_pareto = True
        uniform_draws = pareto_uniform_draws 
        this_alpha = alpha_X1
        use_uniform = False
        use_cauchy = False
  if force_uniform:
        use_pareto = False
        use_uniform = True
        use_cauchy = False
  if force_cauchy:
        use_pareto = False
        use_uniform = False
        use_cauchy = True
        
  if verbose:
        print("X1 : use_pareto = ", use_pareto)
        print("X1 : use_uniform = ", use_uniform)
        print("X1 : use_cauchy = ", use_cauchy)
        
  X1 = generate_data_manifold(N, 
                              d1, 
                              this_alpha, 
                              uniform_draws,
                              use_pareto,
                              use_uniform,
                              use_cauchy)
  dim_X1 = X1.shape[1]
  pp_dim_X1 = calculate_PatnaikPearson_dim(X1)
  nu_over_d_X1 = pp_dim_X1 / dim_X1
  actual_alpha_X1 = 0.0
  if force_pareto:
    actual_alpha_X1 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1, dim_X1)
    
  if verbose:
      print("dim_X1 = ", dim_X1)
      print("pp_dim_X1 = ", pp_dim_X1)
      print("nu_over_d_X1 = ", nu_over_d_X1)
      if force_pareto:
          print("actual_alpha_X1 = ", actual_alpha_X1)
      print("this_alpha = ", this_alpha)

  #this_random = np.random.randint(0, 3)
  this_random = int(np.random.uniform(0,3))
  print("this_random = ", this_random, this_random == 0, this_random == 1, this_random == 2)
  use_pareto = this_random == 0
  use_uniform = this_random == 1
  use_cauchy = this_random == 2
  print(use_pareto, use_uniform, use_cauchy)
  this_alpha = np.random.uniform(0.1,5.0)

  if force_pareto:
        use_pareto = True
        uniform_draws = pareto_uniform_draws 
        this_alpha = alpha_X2
        use_uniform = False
        use_cauchy = False
  if force_uniform:
        use_pareto = False
        use_uniform = True
        use_cauchy = False
  if force_cauchy:
        use_pareto = False
        use_uniform = False
        use_cauchy = True
        
  if verbose:
        print("X2 : use_pareto = ", use_pareto)
        print("X2 : use_uniform = ", use_uniform)
        print("X2 : use_cauchy = ", use_cauchy)
        
  X2 = generate_data_manifold(N, 
                              d2, 
                              this_alpha, 
                              uniform_draws,
                              use_pareto,
                              use_uniform,
                              use_cauchy)
  dim_X2 = X2.shape[1]
  pp_dim_X2 = calculate_PatnaikPearson_dim(X2)
  nu_over_d_X2 = pp_dim_X2 / dim_X2
  actual_alpha_X2 = 0.0
  if force_pareto:
    actual_alpha_X2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X2, dim_X2)
    
  if verbose:
      print("dim_X2 = ", dim_X2)
      print("pp_dim_X2 = ", pp_dim_X2)
      print("nu_over_d_X2 = ", nu_over_d_X2)
      if force_pareto:
          print("actual_alpha_X2 = ", actual_alpha_X2)
      print("this_alpha = ", this_alpha)

  X1concatX2 = np.concatenate([X1, X2], axis=1)
  dim_X1concatX2 = X1concatX2.shape[1]
  pp_dim_X1concatX2 = calculate_PatnaikPearson_dim(X1concatX2)
  nu_over_d_X1concatX2 = pp_dim_X1concatX2 / dim_X1concatX2
  actual_alpha_X1concatX2 = 0.0
  if force_pareto:
    actual_alpha_X1concatX2 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1concatX2, dim_X1concatX2)
    
  if verbose:
      print("dim_X1concatX2 = ", dim_X1concatX2)
      print("pp_dim_X1concatX2 = ", pp_dim_X1concatX2)
      print("nu_over_d_X1concatX2 = ", nu_over_d_X1concatX2)
      if force_pareto:
          print("actual_alpha_X1concatX2 = ", actual_alpha_X1concatX2)

  results_dict = { 
    "actual_alpha_X1" : actual_alpha_X1, 
    "actual_alpha_X2" : actual_alpha_X2, 
    "actual_alpha_X1concatX2" : actual_alpha_X1concatX2, 
    "pp_dim_X1" : pp_dim_X1,
    "pp_dim_X2" : pp_dim_X2,
    "pp_dim_X1concatX2" : pp_dim_X1concatX2,
    "nu_over_d_X1" : nu_over_d_X1,
    "nu_over_d_X2" : nu_over_d_X2, 
    "nu_over_d_X1concatX2" : nu_over_d_X1concatX2
  }
  
  return results_dict
   
def analyse_embeddings(these_embeddings : np.ndarray,
                       verbose : bool = False
                       ):
                           
  print("these_embeddings.shape = ", these_embeddings.shape)
  num_embeddings, embedding_dim = these_embeddings.shape
  print("num_embeddings = ", num_embeddings)
  embedding_norms = np.zeros(num_embeddings)
  for i in range(num_embeddings):
    embedding_norms[i] = math.sqrt(np.dot(these_embeddings[i],these_embeddings[i]))
  print("raw embedding norm stats:")
  display_stats(embedding_norms)

  avg_embedding = np.mean(these_embeddings, axis=0)
  print("norm(avg_embedding) :", math.sqrt(np.dot(avg_embedding,avg_embedding)))

  # demean:
  demeaned_embeddings = these_embeddings - avg_embedding
  sanity_check = np.mean(demeaned_embeddings, axis=0)
  print("sanity check - should be zero : ", np.dot(sanity_check,sanity_check))

  demeaned_embedding_norms = np.zeros(num_embeddings)
  for i in range(num_embeddings):
    demeaned_embedding_norms[i] = math.sqrt(np.dot(demeaned_embeddings[i],demeaned_embeddings[i]))
  print("raw demeaned_embedding norm stats:")
  display_stats(demeaned_embedding_norms)

  N = 2000
  num_iterations = int(num_embeddings / N)
  if verbose:
    print("num_iterations = ", num_iterations)
  if num_embeddings > N:
    print("num_iterations = ", num_iterations)
    these_indices = np.arange(num_embeddings)
    pp_dim_initial_vals = np.zeros(num_iterations)
    pp_dim_demeaned_vals = np.zeros(num_iterations)
    for i in range(num_iterations):
      sample_indices = np.random.choice(these_indices, size=N, replace=False)

      sample_embeddings = these_embeddings[sample_indices,:]
      pp_dim_initial = calculate_PatnaikPearson_dim(sample_embeddings, verbose=False)
      print(i, "pp_dim_initial = ", pp_dim_initial)
      pp_dim_initial_vals[i] = pp_dim_initial

      sample_demeaned_embeddings = demeaned_embeddings[sample_indices,:]
      pp_dim_demeaned = calculate_PatnaikPearson_dim(sample_demeaned_embeddings, verbose=False)
      print(i, "pp_dim_demeaned = ", pp_dim_demeaned)
      pp_dim_demeaned_vals[i] = pp_dim_demeaned

    np_pp_dim_initial_vals = np.array(pp_dim_initial_vals)
    print("raw initial embedding PP dim stats:")
    print("Sample size = ", N, ", num_iterations = ", num_iterations)
    display_stats(np_pp_dim_initial_vals)
    np_pp_dim_demeaned_vals = np.array(pp_dim_demeaned_vals)

    print("raw demeaned embedding PP dim stats:")
    print("Sample size = ", N, ", num_iterations = ", num_iterations)
    display_stats(np_pp_dim_demeaned_vals)
  else:
    pp_dim_embeddings = calculate_PatnaikPearson_dim(these_embeddings, verbose=False)
    pp_dim_demeaned_embeddings = calculate_PatnaikPearson_dim(demeaned_embeddings, verbose=False)
    print("pp_dim_embeddings = ", pp_dim_embeddings)
    print("pp_dim_demeaned_embeddings = ", pp_dim_demeaned_embeddings)
    
def plot_per_layer_graphs(df : pd.DataFrame, 
                          num_layers : int,
                          this_key : str, 
                          this_title : str, 
                          this_image_name : str):

    this_df = df.loc[df['value_type'] == this_key]
    layer_idx_vals = np.arange(0,num_layers,1)
    layer_mean_vals = np.zeros(num_layers)
    layer_std_vals = np.zeros(num_layers)

    for layer_idx in layer_idx_vals:
        df_this_layer = (this_df.loc[this_df['layer_index'] == layer_idx])["values"]
        layer_mean = df_this_layer.mean()
        layer_std = df_this_layer.std()
        layer_mean_vals[layer_idx] = layer_mean
        layer_std_vals[layer_idx] = layer_std

    plt.plot(layer_idx_vals, layer_mean_vals, label = "mean")
    plt.plot(layer_idx_vals, layer_std_vals, label = "std")
    plt.xlabel("layer id")
    plt.legend()
    plt.title(this_title)
    plt.savefig(this_image_name, dpi=300, bbox_inches='tight')
    plt.show()
    
def bert_token_embedding_layerwise_pp_dim_experiment(model : 'transformers.models.bert.modeling_bert.BertModel',  
                                                     tokenizer : 'transformers.models.bert.tokenization_bert.BertTokenizer',
                                                     DEVICE : 'torch.device',
                                                     N : int = 512, 
                                                     num_iterations : int = 1
                                                    ) -> dict:

    max_N = 512 # BERT context length
    N = min(N, max_N)

    valid_bert_base_token_ids = get_valid_bert_base_token_ids()
    vocab_size = len(valid_bert_base_token_ids)
    print("vocab_size = ", vocab_size)
    attention_mask = torch.ones(1, N, dtype=torch.long).to(DEVICE)     # no padding

    layer_pp_dim = {}
    layer_nu_over_d = {}
    layer_nu_over_N = {}
    num_layers = 12
    for layer_id in range(0,num_layers + 1):
        layer_pp_dim[layer_id] = np.zeros(num_iterations)
        layer_nu_over_d[layer_id] = np.zeros(num_iterations)
        layer_nu_over_N[layer_id] = np.zeros(num_iterations)

    for i in range(num_iterations):
        print("iteration " + str(i))
        random_indices = np.random.choice(vocab_size, N, replace = False)
        np_random_token_ids = np.zeros(len(random_indices))
        count = 0
        for ridx in random_indices:
            np_random_token_ids[count] = valid_bert_base_token_ids[ridx]
            count+=1
        print("np_random_token_ids : max = ", np.max(np_random_token_ids), ", min = ", np.min(np_random_token_ids) , 
              ", mean = ", np.mean(np_random_token_ids), ", std = ", np.std(np_random_token_ids))
        
        random_token_ids = torch.tensor(np_random_token_ids, dtype=torch.long)

        input_ids = random_token_ids.unsqueeze(0).to(DEVICE)          # (1, N)
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract per-layer representations
        # hidden_states is a tuple of (num_layers + 1) tensors, each shape (1, N, 768)
        # Index 0 = embedding layer, indices 1–12 = transformer layers 1–12
        hidden_states = outputs.hidden_states  # tuple of 13 tensors
        layer_representations = {}
        for layer_idx, hs in enumerate(hidden_states):
            rep = hs.squeeze(0)
            np_rep = rep.cpu().numpy()
            dim_rep = np_rep.shape[1]
            pp_dim_rep = float(calculate_PatnaikPearson_dim(np_rep))
            nu_over_d_rep = pp_dim_rep / dim_rep
            nu_over_N_rep = pp_dim_rep / N
            layer_pp_dim[layer_idx][i] = pp_dim_rep
            layer_nu_over_d[layer_idx][i] = nu_over_d_rep
            layer_nu_over_N[layer_idx][i] = nu_over_N_rep

    results_dict = {
            "layer_pp_dim" : layer_pp_dim,
            "layer_nu_over_d" : layer_nu_over_d,
            "layer_nu_over_N" : layer_nu_over_N
        }

    return results_dict
    
def DeepSeek_R1_Distill_Qwen_1_5B_token_embedding_layerwise_pp_dim_experiment(model : 'transformers.models.qwen2.modeling_qwen2.Qwen2Model', 
                                                     tokenizer : 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast',
                                                     DEVICE : 'torch.device',
                                                     N : int = 2000, 
                                                     num_iterations : int = 1
                                                    ) -> dict:

    max_N = 130000 # max context length (approx)
    N = min(N, max_N)

    vocab_size = tokenizer.vocab_size
    print("vocab_size = ", vocab_size)
    
    attention_mask = torch.ones(1, N, dtype=torch.long).to(DEVICE)     # no padding

    layer_pp_dim = {}
    layer_nu_over_d = {}
    layer_nu_over_N = {}
    num_layers = 29
    for layer_id in range(0,num_layers + 1):
        layer_pp_dim[layer_id] = np.zeros(num_iterations)
        layer_nu_over_d[layer_id] = np.zeros(num_iterations)
        layer_nu_over_N[layer_id] = np.zeros(num_iterations)

    for i in range(num_iterations):
        print("iteration " + str(i))
        random_token_ids = torch.randperm(vocab_size, dtype=torch.long)[:N]

        input_ids = random_token_ids.unsqueeze(0).to(DEVICE)          # (1, N)
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract per-layer representations
        # hidden_states: tuple of (num_layers + 1) tensors, each shape (1, N, 1536)
        hidden_states = outputs.hidden_states  # tuple of tensors
        layer_representations = {}
        for layer_idx, hs in enumerate(hidden_states):
            rep = hs.squeeze(0)
            np_rep = rep.cpu().numpy().astype(np.float32)
            dim_rep = np_rep.shape[1]
            pp_dim_rep = float(calculate_PatnaikPearson_dim(np_rep))
            nu_over_d_rep = pp_dim_rep / dim_rep
            nu_over_N_rep = pp_dim_rep / N
            layer_pp_dim[layer_idx][i] = pp_dim_rep
            layer_nu_over_d[layer_idx][i] = nu_over_d_rep
            layer_nu_over_N[layer_idx][i] = nu_over_N_rep

    results_dict = {
            "layer_pp_dim" : layer_pp_dim,
            "layer_nu_over_d" : layer_nu_over_d,
            "layer_nu_over_N" : layer_nu_over_N
        }

    return results_dict
    
def normalisation_experiment(N : int,
                            d : int,
                            alpha_X : float,
                            uniform_draws : bool = False,
                            use_pareto : bool = True,
                            use_uniform : bool = False,
                            use_cauchy : bool = False,
                            verbose : bool = False
                            ) -> dict:


  X = generate_data_manifold(N, d, alpha_X, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose)

  dim_X = X.shape[1]
  pp_dim_X = calculate_PatnaikPearson_dim(X)
  nu_over_d_X = pp_dim_X / dim_X
  actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X, dim_X)

  eps = 1e-6
  num_rows = X.shape[0]
  Xnormalised = np.zeros(X.shape)
  for row_num in range(0,num_rows):
    this_row_vector = X[row_num,:]
    this_row_vector_demeaned = this_row_vector - this_row_vector.mean()
    this_norm_squared = (this_row_vector_demeaned**2).sum()
    this_normalisation_factor = 1.0 / math.sqrt(this_norm_squared + eps)
    this_row_vector_normalised = this_row_vector_demeaned * this_normalisation_factor
    Xnormalised[row_num,:] = this_row_vector_normalised

  pp_dim_Xnormalised = calculate_PatnaikPearson_dim(Xnormalised)
  dim_Xnormalised = Xnormalised.shape[1]
  nu_over_d_Xnormalised = pp_dim_Xnormalised / dim_Xnormalised
  actual_alpha_Xnormalised = calculate_alpha_given_nu_over_d_and_d(nu_over_d_Xnormalised, dim_Xnormalised)
  
  results_dict = {
    "pp_dim_X" : pp_dim_X,
    "nu_over_d_X" : nu_over_d_X,
    "actual_alpha_X" : actual_alpha_X,
    "pp_dim_Xnormalised" : pp_dim_Xnormalised, 
    "nu_over_d_Xnormalised" : nu_over_d_Xnormalised,
    "actual_alpha_Xnormalised" : actual_alpha_Xnormalised
  }
  
  return results_dict
  
def DeepSeek_R1_Distill_Qwen_1_5B_token_embedding_initial_pp_dim_experiment(model : 'transformers.models.qwen2.modeling_qwen2.Qwen2Model', 
                                                     tokenizer : 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast',
                                                     DEVICE : 'torch.device',
                                                     N : int = 2000, 
                                                     num_iterations : int = 1,
                                                     verbose : bool = False
                                                    ) -> dict:

    pp_dim_vals = np.zeros(num_iterations)
    nu_over_d_vals = np.zeros(num_iterations)

    max_N = 130000 # max context length (approx)
    N = min(N, max_N)

    vocab_size = tokenizer.vocab_size
    print("vocab_size = ", vocab_size)

    # Extract embeddings directly from the embedding layer
    # No need for a forward pass — just index into the embedding matrix directly
    embedding_layer = model.embed_tokens  # (vocab_size, hidden_size)
    
    for i in range(num_iterations):
        print("iteration " + str(i))
        #random_token_ids = torch.randperm(vocab_size, dtype=torch.long)[:N]
        perm = torch.randperm(vocab_size)[:N]  # random permutation, take first N
        sampled_token_ids = perm.to(DEVICE)
        
        # Extract embeddings directly from the embedding layer
        # No need for a forward pass — just index into the embedding matrix directly
        with torch.no_grad():
            token_embeddings = embedding_layer(sampled_token_ids)  # (N, hidden_size)

        # convert to numpy array
        X = ((token_embeddings.cpu()).numpy()).astype(np.float32)
        # np_rep = (rep.cpu().numpy()).astype(np.float32)
        pp_dim_X = float(calculate_PatnaikPearson_dim(X))
        dim_X = X.shape[1]
        nu_over_d_X = pp_dim_X / dim_X
        if verbose:
            print(i, pp_dim_X, nu_over_d_X)

        pp_dim_vals[i] = pp_dim_X
        nu_over_d_vals[i] = nu_over_d_X

    results_dict = {
        "pp_dim_vals" : pp_dim_vals,
        "nu_over_d_vals" : nu_over_d_vals
    }

    return results_dict
    
def interpolation_experiment(N : int, 
                            d : int, 
                            initial_alpha_X0 : float, 
                            initial_alpha_X1 : float
                            ) -> dict:

	t_vals = np.arange(0.0,1.01,0.01)
	num_iterations = len(t_vals)

	initial_alpha_X0_vals = np.zeros(num_iterations)
	initial_alpha_X1_vals = np.zeros(num_iterations)
	actual_alpha_X0_vals = np.zeros(num_iterations)
	actual_alpha_X1_vals = np.zeros(num_iterations)
	actual_alpha_Xt_vals = np.zeros(num_iterations)
	nu_over_d_X0_vals = np.zeros(num_iterations)
	nu_over_d_X1_vals = np.zeros(num_iterations)
	nu_over_d_Xt_vals = np.zeros(num_iterations)
	estimate_alpha_Xt_vals = np.zeros(num_iterations)
	estimate_nu_over_d_Xt_vals = np.zeros(num_iterations)

	X0 = generate_data_manifold(N, d, initial_alpha_X0)
	pp_dim_X0 = calculate_PatnaikPearson_dim(X0)
	dim_X0 = X0.shape[1]
	nu_over_d_X0 = pp_dim_X0 / dim_X0
	actual_alpha_X0 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X0, dim_X0)

	X1 = generate_data_manifold(N, d, initial_alpha_X1)
	pp_dim_X1 = calculate_PatnaikPearson_dim(X1)
	dim_X1 = X1.shape[1]
	nu_over_d_X1 = pp_dim_X1 / dim_X1
	actual_alpha_X1 = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X1, dim_X1)

	i = 0
	for t in t_vals:
		Xt = (1.0 - t) * X0 + t * X1
		pp_dim_Xt = calculate_PatnaikPearson_dim(Xt)
		dim_Xt = Xt.shape[1]
		nu_over_d_Xt = pp_dim_Xt / dim_Xt
		actual_alpha_Xt = calculate_alpha_given_nu_over_d_and_d(nu_over_d_Xt, dim_Xt)
		estimate_alpha_Xt = (1.0 - t) * actual_alpha_X0 + t * actual_alpha_X1
		_, estimate_nu_over_d_Xt = calculate_nu_and_nu_over_d_given_alpha_d_analytic(estimate_alpha_Xt, dim_Xt)

		print("actual_alpha_X0 = ", actual_alpha_X0, ", actual_alpha_X1 = ", actual_alpha_X1, "t = ", t, ", actual_alpha_Xt = ", actual_alpha_Xt, "estimate_alpha_Xt = ", estimate_alpha_Xt)

		initial_alpha_X0_vals[i] = initial_alpha_X0
		initial_alpha_X1_vals[i] = initial_alpha_X1
		actual_alpha_X0_vals[i] = actual_alpha_X0
		actual_alpha_X1_vals[i] = actual_alpha_X1
		actual_alpha_Xt_vals[i] = actual_alpha_Xt
		nu_over_d_X0_vals[i] = nu_over_d_X0
		nu_over_d_X1_vals[i] = nu_over_d_X1
		nu_over_d_Xt_vals[i] = nu_over_d_Xt
		estimate_alpha_Xt_vals[i] = estimate_alpha_Xt
		estimate_nu_over_d_Xt_vals[i] = estimate_nu_over_d_Xt
		i += 1
  
	results_dict = {
        "t_vals" : t_vals,
		"initial_alpha_X0_vals" : initial_alpha_X0_vals,
		"initial_alpha_X1_vals" : initial_alpha_X1_vals,
		"actual_alpha_X0_vals" : actual_alpha_X0_vals,
		"actual_alpha_X1_vals" : actual_alpha_X1_vals,
		"actual_alpha_Xt_vals" : actual_alpha_Xt_vals,
		"nu_over_d_X0_vals" : nu_over_d_X0_vals,
		"nu_over_d_X1_vals" : nu_over_d_X1_vals,
		"nu_over_d_Xt_vals" : nu_over_d_Xt_vals,
		"estimate_alpha_Xt_vals" : estimate_alpha_Xt_vals,
		"estimate_nu_over_d_Xt_vals" : estimate_nu_over_d_Xt_vals
	}

	return results_dict
    
def softmax_experiment(	N : int, 
						d : int, 
						temps : np.ndarray, 
						alpha : float, 
						uniform_draws : bool = False,
						use_pareto : bool = True,
						use_uniform : bool = False,
						use_cauchy : bool = False,
						verbose : bool = False
						) -> dict:



	X = generate_data_manifold(N, d, alpha, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose)

	pp_dim_X = calculate_PatnaikPearson_dim(X)
	dim_X = X.shape[1]
	nu_over_d_X = pp_dim_X / dim_X
	actual_alpha_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_X, dim_X)

	#log_temps = np.arange(-6.0,11.0,1.0) # np.arange(-1.0, 1.1, 0.1)
	#print(log_temps)
	#temps = 10**log_temps
	#print(temps)

	num_iterations = len(temps)

	pp_dim_softmax_X_vals = np.zeros(num_iterations)
	nu_over_d_softmax_X_vals = np.zeros(num_iterations)
	actual_alpha_softmax_X_vals = np.zeros(num_iterations)

	i = 0
	for T in temps:
		softmax_X = row_wise_softmax(X,T)
		pp_dim_softmax_X = calculate_PatnaikPearson_dim(softmax_X)
		dim_softmax_X = softmax_X.shape[1]
		nu_over_d_softmax_X = pp_dim_softmax_X / dim_softmax_X
		actual_alpha_softmax_X = calculate_alpha_given_nu_over_d_and_d(nu_over_d_softmax_X, dim_softmax_X)

		pp_dim_softmax_X_vals[i] = pp_dim_softmax_X
		nu_over_d_softmax_X_vals[i] = nu_over_d_softmax_X
		actual_alpha_softmax_X_vals[i] = actual_alpha_softmax_X

		print(i, T, pp_dim_softmax_X, nu_over_d_softmax_X, actual_alpha_softmax_X)
		i += 1
		
	results_dict = {
		"pp_dim_X" : pp_dim_X,
		"nu_over_d_X" : nu_over_d_X,
		"actual_alpha_X" : actual_alpha_X,
		"pp_dim_softmax_X_vals" : pp_dim_softmax_X_vals,
		"nu_over_d_softmax_X_vals" : nu_over_d_softmax_X_vals,
		"actual_alpha_softmax_X_vals" : actual_alpha_softmax_X_vals
	}
	
	return results_dict
    

def pp_dim_AB_experiment(num_iterations : int = 10, size_scale : int = 500) -> dict:

	these_alphas = np.arange(0.1, 5.15, 0.05) # 100

	uniform_draws = True
	use_pareto = True
	use_uniform = False
	use_cauchy = False
	verbose = False
	use_svd = True

	pp_dim_A_vals = np.zeros(num_iterations)
	pp_dim_B_vals = np.zeros(num_iterations)
	pp_dim_AB_vals = np.zeros(num_iterations)
	min_pp_dim_A_pp_dim_B_vals = np.zeros(num_iterations)
	max_pp_dim_A_pp_dim_B_vals = np.zeros(num_iterations)

	nu_over_d_A_vals = np.zeros(num_iterations)
	nu_over_d_B_vals = np.zeros(num_iterations)
	nu_over_d_AB_vals = np.zeros(num_iterations)

	min_nu_over_d_A_nu_over_d_B_vals = np.zeros(num_iterations)
	max_nu_over_d_A_nu_over_d_B_vals = np.zeros(num_iterations)
	nu_over_d_A_times_nu_over_d_B_vals = np.zeros(num_iterations)

	for i in range(num_iterations):

		N = int(size_scale * ( 1 + np.random.uniform(0,1)))
		d = N + int(size_scale * (np.random.uniform(0,1) - 0.5))
		m = d + int(size_scale * np.random.uniform(0,1))
		print(i, N, d, m)

		alpha_1 = random.choice(these_alphas)
		alpha_2 = random.choice(these_alphas)
		print(i, alpha_1, alpha_2)

		A = generate_data_manifold(N, d, alpha_1, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose, use_svd)
		B = generate_data_manifold(d, m, alpha_2, uniform_draws, use_pareto, use_uniform, use_cauchy, verbose, use_svd)

		pp_dim_A = calculate_PatnaikPearson_dim(A)
		dim_A = A.shape[1]
		nu_over_d_A = pp_dim_A / dim_A

		pp_dim_B = calculate_PatnaikPearson_dim(B)
		dim_B = B.shape[1]
		nu_over_d_B = pp_dim_B / dim_B

		cp_AB = cp.matmul(cp.array(A), cp.array(B))
		AB = cp.asnumpy(cp_AB)
		pp_dim_AB = calculate_PatnaikPearson_dim(AB)
		dim_AB = AB.shape[1]
		nu_over_d_AB = pp_dim_AB / dim_AB

		min_pp_dim_A_pp_dim_B = min(pp_dim_A, pp_dim_B)
		max_pp_dim_A_pp_dim_B = max(pp_dim_A, pp_dim_B)

		min_nu_over_d_A_nu_over_d_B = min(nu_over_d_A, nu_over_d_B)
		max_nu_over_d_A_nu_over_d_B = max(nu_over_d_A, nu_over_d_B)

		nu_over_d_A_times_nu_over_d_B = nu_over_d_A * nu_over_d_B

		pp_dim_A_vals[i] = pp_dim_A
		pp_dim_B_vals[i] = pp_dim_B
		pp_dim_AB_vals[i] = pp_dim_AB
		min_pp_dim_A_pp_dim_B_vals[i] = min_pp_dim_A_pp_dim_B
		max_pp_dim_A_pp_dim_B_vals[i] = max_pp_dim_A_pp_dim_B

		nu_over_d_A_vals[i] = nu_over_d_A
		nu_over_d_B_vals[i] = nu_over_d_B
		nu_over_d_AB_vals[i] = nu_over_d_AB

		min_nu_over_d_A_nu_over_d_B_vals[i] = min_nu_over_d_A_nu_over_d_B
		max_nu_over_d_A_nu_over_d_B_vals[i] = max_nu_over_d_A_nu_over_d_B
		nu_over_d_A_times_nu_over_d_B_vals[i] = nu_over_d_A * nu_over_d_B

		print(i, pp_dim_AB, pp_dim_A, pp_dim_B)
		print(i, "is PP(AB) = ", pp_dim_AB , " leq min(PP(A), PP(B)) = ", min(pp_dim_A, pp_dim_B))
		print(i, "is nu/d (AB) = ", nu_over_d_AB, " leq min( nu/d (A), nu/d (B)) = ", min(nu_over_d_A, nu_over_d_B)) 
		print(i, "is nu/d (A) * nu/d (B) = ", nu_over_d_A * nu_over_d_B, " leq  nu/d (AB) = ", nu_over_d_AB)
		
	results_dict = {
	 "pp_dim_A_vals" : pp_dim_A_vals,
	 "pp_dim_B_vals" : pp_dim_B_vals,
	 "pp_dim_AB_vals" : pp_dim_AB_vals,
	 "min_pp_dim_A_pp_dim_B_vals" : min_pp_dim_A_pp_dim_B_vals,
	 "max_pp_dim_A_pp_dim_B_vals" : max_pp_dim_A_pp_dim_B_vals,
	 "nu_over_d_A_vals" : nu_over_d_A_vals,
	 "nu_over_d_B_vals" : nu_over_d_B_vals,
	 "nu_over_d_AB_vals" : nu_over_d_AB_vals,
	 "min_nu_over_d_A_nu_over_d_B_vals" : min_nu_over_d_A_nu_over_d_B_vals,
	 "max_nu_over_d_A_nu_over_d_B_vals" : max_nu_over_d_A_nu_over_d_B_vals,
	 "nu_over_d_A_times_nu_over_d_B_vals" : nu_over_d_A_times_nu_over_d_B_vals
	}
	
	return results_dict
        
        

    
