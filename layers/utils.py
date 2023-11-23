def no_op(*args, **kwargs):
    pass


class NoOpContextManager:

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        return
