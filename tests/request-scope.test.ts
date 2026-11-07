import assert from "node:assert/strict";
import test from "node:test";
import { createRequestScope } from "../frontend/src/requestScope.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

test("a late response cannot replace the current filter's rows", async () => {
  const scope = createRequestScope();
  const first = deferred<string[]>(),
    second = deferred<string[]>();
  let rows: string[] = [],
    loading = true;
  const callbacks = {
    success: (value: string[]) => {
      rows = value;
    },
    error: assert.fail,
    settled: () => {
      loading = false;
    },
  };
  const old = scope.run(() => first.promise, callbacks);
  const current = scope.run(() => second.promise, callbacks);
  second.resolve(["new-filter"]);
  await current;
  first.resolve(["old-filter"]);
  await old;
  assert.deepEqual(rows, ["new-filter"]);
  assert.equal(loading, false);
});

test("old pagination cannot append into a replacement query", async () => {
  const scope = createRequestScope();
  const olderPage = deferred<string[]>(),
    replacement = deferred<string[]>();
  let rows = ["initial-filter"];
  const old = scope.run(() => olderPage.promise, {
    success: (value) => {
      rows = [...rows, ...value];
    },
    error: assert.fail,
  });
  const current = scope.run(() => replacement.promise, {
    success: (value) => {
      rows = value;
    },
    error: assert.fail,
  });
  replacement.resolve(["replacement-filter"]);
  await current;
  olderPage.resolve(["older-wrong-filter"]);
  await old;
  assert.deepEqual(rows, ["replacement-filter"]);
});

test("stale errors and finally cannot overwrite current errors or loading", async () => {
  const scope = createRequestScope();
  const first = deferred<string>(),
    second = deferred<string>();
  let error: unknown = null,
    loading = true;
  const callbacks = {
    success: () => {},
    error: (value: unknown) => {
      error = value;
    },
    settled: () => {
      loading = false;
    },
  };
  const old = scope.run(() => first.promise, callbacks);
  const current = scope.run(() => second.promise, callbacks);
  first.reject(new Error("superseded failure"));
  await old;
  assert.equal(error, null);
  assert.equal(loading, true);
  second.reject(new Error("current failure"));
  await current;
  assert.equal((error as Error).message, "current failure");
  assert.equal(loading, false);
});

test("refresh invalidates a plan verification before its response arrives", async () => {
  const verification = createRequestScope();
  const oldResult = deferred<string>();
  const visible: string[] = [];
  let settled = 0;
  const old = verification.run(() => oldResult.promise, {
    success: (value) => visible.push(value),
    error: assert.fail,
    settled: () => {
      settled++;
    },
  });
  verification.invalidate(); // refreshing the plan invalidates its old check immediately
  oldResult.resolve("valid-for-plan-A");
  await old;
  assert.deepEqual(visible, []);
  assert.equal(settled, 0);
  await verification.run(async () => "valid-for-plan-B", {
    success: (value) => visible.push(value),
    error: assert.fail,
    settled: () => {
      settled++;
    },
  });
  assert.deepEqual(visible, ["valid-for-plan-B"]);
  assert.equal(settled, 1);
});

test("unmount invalidation ignores both a late failure and its settled callback", async () => {
  const scope = createRequestScope(),
    response = deferred<string>();
  const pending = scope.run(() => response.promise, {
    success: () => assert.fail("unmounted success"),
    error: () => assert.fail("unmounted error"),
    settled: () => assert.fail("unmounted finally"),
  });
  scope.invalidate();
  response.reject(new Error("late failure"));
  await pending;
});

test("current synchronous failures still report errors and settle", async () => {
  const scope = createRequestScope();
  const events: string[] = [];
  await scope.run(
    () => {
      throw new Error("current synchronous failure");
    },
    {
      success: () => assert.fail("unexpected success"),
      error: (error) => events.push((error as Error).message),
      settled: () => events.push("settled"),
    },
  );
  assert.deepEqual(events, ["current synchronous failure", "settled"]);
});
