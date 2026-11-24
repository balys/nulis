"""The Nulis rig: the evaluator side of the project.

Everything here sits outside the three loops (notes, harness, weights). It defines the tasks,
runs rollouts in a sandbox, grades them, and records the result against the model, adapter and
harness version that produced it. The harness under test is a separate program (kilpa) that
the rig starts as a command. Nothing in this package is part of the editable harness surface.
"""
