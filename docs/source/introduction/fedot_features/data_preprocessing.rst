Data preprocessing
------------------

FEDOT uses two types of preprocessing: obligatory and optional.

The first one, as you might guess, cares about something that can break
your program, these are: inf in features, nan-features, nan targets,
extra spaces in categorical columns.

The second one depends on composed pipeline structure, and is applied only if
it is necessary for the next model in a processing queue.