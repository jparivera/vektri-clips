/** Only the newest request in a scope may update data, errors or loading state. */
export function createRequestScope() {
  let generation = 0;
  return {
    invalidate() {
      generation++;
    },
    async run<T>(
      operation: () => Promise<T>,
      callbacks: {
        success: (value: T) => void;
        error: (error: unknown) => void;
        settled?: () => void;
      },
    ): Promise<void> {
      const request = ++generation;
      try {
        const value = await operation();
        if (request === generation) callbacks.success(value);
      } catch (error) {
        if (request === generation) callbacks.error(error);
      } finally {
        if (request === generation) callbacks.settled?.();
      }
    },
  };
}
