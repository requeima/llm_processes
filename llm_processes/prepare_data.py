import pickle
import numpy as np


from .helpers import scale_y, sort_test_by_distance_from_train, get_dimension, sequential_sort, randomize


def prepare_data(args):
    # load the data
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    results = {'data': data, 'args': args}

    results['dim_x'] = get_dimension(results['data']['x_train'])
    results['dim_y'] = get_dimension(results['data']['y_train'])

    if (args.y_min is not None) and (args.y_max is not None):  # scale the ys to the new range
        old_min = np.min(results['data']['y_true'])
        old_max = np.max(results['data']['y_true'])
        results['data']['y_true'] = scale_y(ys=np.array(results['data']['y_true']), old_min=old_min, old_max=old_max, new_min=args.y_min, new_max=args.y_max)
        results['data']['y_train'] = scale_y(ys=np.array(results['data']['y_train']), old_min=old_min, old_max=old_max, new_min=args.y_min, new_max=args.y_max)
        results['data']['y_test'] = scale_y(ys=np.array(results['data']['y_test']), old_min=old_min, old_max=old_max, new_min=args.y_min, new_max=args.y_max)

    # sort the train and test sets
    x_ordering = results['data']['x_ordering'] if 'x_ordering' in results['data'] else None
    if args.forecast:
        assert results['dim_x'] == 1  # this is for time series forcasting only
        args.prompt_ordering = 'sequential'  # force training order to be sequential

    # train: We sort the training points accoring to the prompt ordering option.
    # We do 'random' here. Distance and sequential sorting is performed in construct_prompts.
    if args.prompt_ordering == 'random':
        results['data']['x_train'], results['data']['y_train'] = randomize(
                    x=results['data']['x_train'],
                    y=results['data']['y_train']
        )

    # test: Sort order for I-LLMP does not matter. Only sort if auutoregressive (A-LLMP).
    if args.autoregressive:
        if args.forecast:
            assert results['dim_x'] == 1  # this is for time series forcasting only
            # sort based on x
            results['data']['x_test'], results['data']['y_test'] = sequential_sort(
                x=results['data']['x_test'],
                y=results['data']['y_test'],
                x_ordering=x_ordering
            )
        else:
            if args.sort_x_test and (x_ordering is not None):  # don't do this if xs are not numerical
                # sort test points by distance to the train points.
                results['data']['x_test'], results['data']['y_test'] =\
                    sort_test_by_distance_from_train(
                        results['data']['x_train'],
                        results['data']['x_test'],
                        results['data']['y_test']
                    )
            else:
                # randomize test points
                permutation = np.random.permutation(len(results['data']['x_test']))
                results['data']['x_test'] = results['data']['x_test'][permutation]
                results['data']['y_test'] = np.array(results['data']['y_test'])[permutation]

    return results