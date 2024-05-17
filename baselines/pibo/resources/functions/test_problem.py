class TestProblem:
    def __init__(self, problem, maximise=True):
        self.problem = problem
        self.name = self.problem.__class__.__name__
        self.dim = self.problem.dim
        self.input_columns = []
        for i in range(self.dim):
            self.input_columns.append(f"x{i}")
        self.bound = {}
        if isinstance(self.problem.lb, float):
            for i in self.input_columns:
                self.bound[i] = (self.problem.lb, self.problem.ub)
        else:
            for i, input_column in enumerate(self.input_columns):
                lb_ = self.problem.lb[i]
                ub_ = self.problem.ub[i]

                self.bound[input_column] = (lb_, ub_)
        self.inverse = -1 if maximise else 1

    def __call__(self, kwargs):
        input_data = list(kwargs.values())
        result = self.problem(input_data)
        if len(result) == 1:
            return result[0] * self.inverse
        return result * self.inverse
