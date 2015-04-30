# functions and data used for debugging purposes

# holds all counts kept when running the program
_count_dict = {}


# creates a new counter with the specified name when first called; increments counter on future calls
# also prints the current count based on the specified frequency
def run_counter(name, frequency):

    # first check if counter already exists
    if name in _count_dict:

        # print count and increment
        count = _count_dict[name]
        if count % frequency == 0:
            print(name + ": " + str(count))
        _count_dict[name] += 1
    # otherwise create new counter
    else:
        print(name + ": 0")
        _count_dict[name] = 1


# resets a counter so that it can be used again
def reset_counter(name):
    if name in _count_dict:
        _count_dict[name] = 0
