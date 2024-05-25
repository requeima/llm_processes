
import torch
import math
import re
import decimal
import numpy as np
import sys


ctx = decimal.Context()
ctx.prec = 20


def _map_to_ordinal(array, ordering):
    if ordering is not None:
        return np.array([ordering[key] for key in array])
    else:
        return array


def scale_y(ys, old_min, old_max, new_min, new_max):
    assert ys.ndim == 1
    return ((ys - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min


def randomize(x, y):
    permutation = np.random.permutation(len(x))
    return  (np.array(x))[permutation], (np.array(y))[permutation]


def sequential_sort(x, y, x_ordering):
    sort_indices = np.argsort(np.array(_map_to_ordinal(x, x_ordering)))
    return (np.array(x))[sort_indices], (np.array(y))[sort_indices]


def sort_test_by_distance_from_train(x_train, x_test, y_test):
    dim_x = x_train.ndim
    distances = []
    for i in x_test:
        min_distance = sys.float_info.max
        for j in x_train: 
            if dim_x > 1:
                distance = math.dist(i, j)
            else:
                distance = abs(i - j)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    distances = np.array(distances)
    sort_indices = np.argsort(distances)
    x_test_sorted = (np.array(x_test))[sort_indices]
    y_test_sorted = (np.array(y_test))[sort_indices]

    return x_test_sorted, y_test_sorted


def get_dimension(a):
    if a.ndim > 1:
        return a.shape[1] # return the second dimension size
    else:
        return 1

def _float_to_str(f, num_decimal=None, add_spaces=False):
    """Convert float to string without resorting to scientific notation."""
    d1 = ctx.create_decimal(repr(f))
    if num_decimal is not None:
        d1 = round(d1, num_decimal)
    s = format(d1, 'f')
    if add_spaces:
        s = (" ".join(s))
    return s 


def floats_to_str(nums, num_decimal, dim=1, add_spaces=False):
    if np.ndim(nums) == 0:
        return _float_to_str(nums, num_decimal, add_spaces)  # when y_dim = 1, only a scalar is passed
    assert len(nums) > 0
    if dim > 1:  # can have multiple dimensions in x and y
        return [[_float_to_str(value, num_decimal, add_spaces) for value in group] for group in nums]
    else:
        return [_float_to_str(num, num_decimal, add_spaces) for num in nums]


def _format_observed_data_point(x, y, dim_x, dim_y, first_prefix, next_prefix, break_str):
    if (dim_x > 1) and (dim_y > 1):
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        y_point_string = ''
        for i in range(dim_y):
            y_point_string += next_prefix
            y_point_string += y[i]
        return f'{x_point_string}{y_point_string}{break_str}'
    elif dim_x > 1:
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        return f'{x_point_string}{next_prefix}{y}{break_str}'
    elif dim_y > 1:
        y_point_string = ''
        for i in range(dim_y):
            y_point_string += next_prefix
            y_point_string += y[i]
        return f'{first_prefix}{x}{y_point_string}{break_str}'
    else: # dim_x = dim_y = 1
       return f'{first_prefix}{x}{next_prefix}{y}{break_str}'
    

def _format_query_data_point(x, dim_x, first_prefix, next_prefix):
    if dim_x > 1:
        x_point_string = ''
        for i in range(dim_x):
            if i == 0:
                x_point_string += first_prefix
            else:
                x_point_string += next_prefix
            x_point_string += x[i]
        return f'{x_point_string}{next_prefix}'
    else: # dim_x = dim_y = 1
       return f'{first_prefix}{x}{next_prefix}'


def construct_prompts(
        x_train,
        y_train,
        x_test,
        prefix='',
        x_prefix='',
        y_prefix=', ',
        break_str='\n',
        remove_space=True,
        dim_x=1,
        dim_y=1,
        num_decimal_x=0,
        num_decimal_y=0,
        order='distance',
        add_spaces=False,
        x_ordering=None
        ):

    # Convert xy train and x test to str.
    if x_ordering is not None:  # xs are already a string
        str_x_train = x_train
        str_x_test = x_test
    else:
        str_x_train = floats_to_str(x_train, num_decimal_x, dim_x, add_spaces)
        str_x_test = floats_to_str(x_test, num_decimal_x, dim_x, add_spaces)
    str_y_train = floats_to_str(y_train, num_decimal_y, dim_y, add_spaces)

    if order == 'random':
        # note:
        # we assume that the input training data is already in random order,
        # so we just need to construct the base prompt here
        base_prompt = prefix
        for x, y in zip(str_x_train, str_y_train):
            base_prompt += _format_observed_data_point(
                x=x,
                y=y,
                dim_x=dim_x,
                dim_y=dim_y,
                first_prefix=x_prefix,
                next_prefix=y_prefix,
                break_str=break_str
            )
    elif order == 'sequential':
            sort_indices = np.argsort(np.array(_map_to_ordinal(x_train, x_ordering)))
            str_x_train_sorted = ((np.array(str_x_train))[sort_indices]).tolist()
            str_y_train_sorted = ((np.array(str_y_train))[sort_indices]).tolist()
            base_prompt = prefix
            for x, y in zip(str_x_train_sorted, str_y_train_sorted):
                base_prompt += _format_observed_data_point(
                    x=x,
                    y=y,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    first_prefix=x_prefix,
                    next_prefix=y_prefix,
                    break_str=break_str
                )

    prompts = []
    for (xt_str, xt_num) in zip(str_x_test, _map_to_ordinal(x_test, x_ordering)):
        if order == 'distance':
            distances = []
            for value in _map_to_ordinal(x_train, x_ordering):
                if dim_x > 1:
                    distances.append(math.dist(xt_num, value))
                else:
                    distances.append(abs(xt_num - value))
            sort_indices = np.flip(np.argsort(distances))
            str_x_train_sorted = ((np.array(str_x_train))[sort_indices]).tolist()
            str_y_train_sorted = ((np.array(str_y_train))[sort_indices]).tolist()

            base_prompt = prefix
            for x, y in zip(str_x_train_sorted, str_y_train_sorted):
                base_prompt += _format_observed_data_point(
                    x=x,
                    y=y,
                    dim_x=dim_x,
                    dim_y=dim_y,
                    first_prefix=x_prefix,
                    next_prefix=y_prefix, break_str=break_str
                )

        prompt = f'{base_prompt}{_format_query_data_point(x=xt_str, dim_x=dim_x, first_prefix=x_prefix, next_prefix=y_prefix)}'
        if remove_space:
            prompt = prompt.rstrip(' ')
        prompts.append(prompt)
    return prompts

def _generate_max_min_values(n, k):
    # Calculate the part before the decimal
    before_decimal = sum(9 * 10**i for i in range(n))
    # Calculate the part after the decimal
    after_decimal = sum(9 * 10**-i for i in range(1, k + 1))
    # Combine both parts
    return before_decimal + after_decimal

def get_num_from_gen(gen, break_str='\n', dim_y=1, max_generated_length=7, num_decimal_places_y=2):
    gen = gen.replace(" ","") # remove any spaces, we add spaces for phi
    nums = re.findall(r'-?\d+\.?\d*', gen)
    
    # if the generataion does not contain any numbers, return None, throw away sample
    if not nums:
        return None
    
    if dim_y > 1:
        # throw away sample if it doesn't contain a break_str
        if break_str not in gen:
             return None
        assert len(nums) >= dim_y
        res = []
        for i in range(dim_y):
            res.append(float(nums[i]))
        res = np.array(res)
    else:
        # determine max and min generated values 
        if num_decimal_places_y == 0:
            max_val = _generate_max_min_values(max_generated_length - 1, 0)
            min_val = -_generate_max_min_values(max_generated_length - 2, 0)
        else:
            max_val = _generate_max_min_values(max_generated_length - num_decimal_places_y - 2, num_decimal_places_y)
            min_val = -_generate_max_min_values(max_generated_length - num_decimal_places_y - 3, num_decimal_places_y)
        
        res = float(nums[0])
        
        if break_str not in gen and "." not in nums[0]:
            if res > max_val:
                res = max_val
            elif res < min_val:    
                res = min_val
            else:
                res = None        
        
    return res


def compute_mse(a, b):
    return np.mean((np.array(a) - np.array(b)) ** 2)


def process_generated_results(gen_results, break_str='\n', dim_y=1, max_generated_length=7, num_decimal_places_y=2):
    # Get all sampled y values. Shape is (num ys, num samples).
    num_xs = len(gen_results['data']['x_test'])
    y_tests = [[] for _ in range(num_xs)]
    y_test_mean = [np.nan for _ in range(num_xs)]
    y_test_median = [np.nan for _ in range(num_xs)]
    y_test_std = [np.nan for _ in range(num_xs)]
    y_test_lower = [np.nan for _ in range(num_xs)]
    y_test_upper = [np.nan for _ in range(num_xs)]
    for i in range(len(gen_results['gen'])):
        if not gen_results['gen'][i]:
            continue
        ys = []
        for j, txt in enumerate(gen_results['gen'][i]):
            y = get_num_from_gen(
                gen=txt,
                break_str=break_str,
                dim_y=dim_y,
                max_generated_length=max_generated_length,
                num_decimal_places_y=num_decimal_places_y
            )
            if y is not None:
                ys.append(y)
        y_tests[i] += ys
        if dim_y > 1:
            ys = np.array(ys)
            y_test_mean[i] = np.mean(ys, axis=0)
            y_test_median[i] = np.median(ys, axis=0)
            y_test_std[i] = np.std(ys, axis=0)
            y_test_lower[i] = np.percentile(ys, 2.5, axis=0)
            y_test_upper[i] = np.percentile(ys, 97.5, axis=0)  
        else:
            y_test_mean[i] = np.mean(ys)
            y_test_median[i] = np.median(ys)
            y_test_std[i] = np.std(ys)
            y_test_lower[i] = np.percentile(ys, 2.5)
            y_test_upper[i] = np.percentile(ys, 97.5)

    if dim_y > 1:
        mae = [np.mean(np.abs((np.array(y_test_median)[:, i] -
                               gen_results['data']['y_test'][:, i]))) 
               for i in range(gen_results['data']['y_test'].shape[1])]
        
        mse = [compute_mse(np.array(y_test_mean)[:, i], 
                           gen_results['data']['y_test'][:, i]) 
               for i in range(gen_results['data']['y_test'].shape[1])]
        
    else:
        mse = compute_mse(
            y_test_mean,
            gen_results['data']['y_test'][: len(y_test_mean)]
        )

        mae = np.mean(np.abs(y_test_median - np.array(gen_results['data']['y_test'])))


    gen_results['y_test'] = y_tests
    if dim_y == 1:  # only used in black box opt with one output y
        gen_results['y_test_max_x'] = gen_results['data']['x_test'][np.argmax(np.max(np.array(y_tests), axis=1))]  # find argmax of the largest sample
    gen_results['y_test_mean'] = y_test_mean
    gen_results['y_test_median'] = y_test_median
    gen_results['y_test_std'] = y_test_std
    gen_results['y_test_lower'] = y_test_lower
    gen_results['y_test_upper'] = y_test_upper
    gen_results['mse'] = mse
    gen_results['mae'] = mae

    print(f'mae: {mae}')
    return gen_results


# this API is in python 3.9, but this is need if running python < 3.9
def my_removesuffix(self: str, suffix: str, /) -> str:
    # suffix='' should not call self[:-0].
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]
