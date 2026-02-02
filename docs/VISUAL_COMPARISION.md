# Visual Comparison: Before vs After Fix

## Architecture Overview

### BEFORE (Broken - Causes Deadlock)

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Thread                          │
│                                                             │
│  1. Submit tasks to ThreadPoolExecutor                      │
│     └─> for q_sample in query_samples                      │
│                                                             │
│  2. Wait for futures as_completed()                         │
│     └─> future.result() → (success, query_id)              │
│                            ↑                                │
│                            │ Only status returned!          │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             │
                             │
┌────────────────────────────┴────────────────────────────────┐
│                    Worker Thread Pool                       │
│                  (10-128 concurrent threads)                │
│                                                             │
│  ┌────────────────────────────────────────────┐            │
│  │ Worker Thread 1                            │            │
│  │                                            │            │
│  │ 1. process_single_async(q_sample)          │            │
│  │    └─> _process_api_single()               │            │
│  │        └─> HTTP Request to API Server      │            │
│  │        └─> _process_single_response()      │            │
│  │            └─> Create numpy array          │            │
│  │            └─> lg.QuerySamplesComplete() ❌│◄───┐       │
│  │                  ↑                          │    │       │
│  │                  └─ PROBLEM: Called from   │    │       │
│  │                     worker thread!          │    │       │
│  └────────────────────────────────────────────┘    │       │
│                                                     │       │
│  ┌────────────────────────────────────────────┐    │       │
│  │ Worker Thread 2                            │    │       │
│  │                                            │    │       │
│  │ 1. process_single_async(q_sample)          │    │       │
│  │    └─> _process_api_single()               │    │       │
│  │        └─> HTTP Request to API Server      │    │       │
│  │        └─> _process_single_response()      │    │       │
│  │            └─> Create numpy array          │    │       │
│  │            └─> lg.QuerySamplesComplete() ❌│◄───┤       │
│  │                  ↑                          │    │       │
│  │                  └─ PROBLEM: Concurrent!   │    │       │
│  └────────────────────────────────────────────┘    │       │
│                                                     │       │
│  ┌────────────────────────────────────────────┐    │       │
│  │ Worker Thread N                            │    │       │
│  │                                            │    │       │
│  │ 1. process_single_async(q_sample)          │    │       │
│  │    └─> _process_api_single()               │    │       │
│  │        └─> HTTP Request to API Server      │    │       │
│  │        └─> _process_single_response()      │    │       │
│  │            └─> Create numpy array          │    │       │
│  │            └─> lg.QuerySamplesComplete() ❌│◄───┘       │
│  │                  ↑                          │            │
│  │                  └─ PROBLEM: Race condition│            │
│  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   MLPerf LoadGen Library     │
              │                              │
              │  Internal synchronization    │
              │  expects single-threaded     │
              │  callback invocations!       │
              │                              │
              │  🔒 DEADLOCK! 💥             │
              └──────────────────────────────┘

PROBLEMS:
❌ Multiple threads call lg.QuerySamplesComplete() concurrently
❌ LoadGen internal locks cause deadlock
❌ Race conditions on self.response_arrays dictionary
❌ Unpredictable thread context for callbacks
```

### AFTER (Fixed - Thread-Safe)

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Thread                          │
│                                                             │
│  1. Submit tasks to ThreadPoolExecutor                      │
│     └─> for q_sample in query_samples                      │
│                                                             │
│  2. Wait for futures as_completed()                         │
│     └─> future.result() → (success, q_sample, response_data)│
│                            ↑                                │
│                            │ Full response data!            │
│                            │                                │
│  3. Extract response data:                                  │
│     └─> query_id = response_data['query_id']               │
│     └─> output_data_ptr = response_data['output_data_ptr'] │
│     └─> output_data_size = response_data['output_data_size']│
│     └─> n_tokens = response_data['n_tokens']               │
│                                                             │
│  4. Create LoadGen response:                                │
│     └─> response_array = [lg.QuerySampleResponse(...)]     │
│                                                             │
│  5. Call LoadGen from MAIN THREAD ONLY: ✅                  │
│     └─> lg.QuerySamplesComplete(response_array)            │
│                                                             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ Returns data dictionary
                             │
┌────────────────────────────┴────────────────────────────────┐
│                    Worker Thread Pool                       │
│                  (10-128 concurrent threads)                │
│                                                             │
│  ┌────────────────────────────────────────────┐            │
│  │ Worker Thread 1                            │            │
│  │                                            │            │
│  │ 1. process_single_async(q_sample)          │            │
│  │    └─> response_data = _process_api_single()│           │
│  │        └─> HTTP Request to API Server      │            │
│  │        └─> return _process_single_response()│           │
│  │            └─> Create numpy array          │            │
│  │            └─> Store in self.response_arrays│           │
│  │            └─> return {                     │            │
│  │                  'query_id': ...,           │            │
│  │                  'output_data_ptr': ...,    │            │
│  │                  'output_data_size': ...,   │            │
│  │                  'n_tokens': ...            │            │
│  │                } ✅                          │            │
│  │                                            │            │
│  │ 2. return (True, q_sample, response_data)  │────┐       │
│  └────────────────────────────────────────────┘    │       │
│                                                     │       │
│  ┌────────────────────────────────────────────┐    │       │
│  │ Worker Thread 2                            │    │       │
│  │                                            │    │       │
│  │ 1. process_single_async(q_sample)          │    │       │
│  │    └─> response_data = _process_api_single()│   │       │
│  │        └─> HTTP Request to API Server      │    │       │
│  │        └─> return _process_single_response()│   │       │
│  │            └─> return {...} ✅              │    ├──────▶│
│  │                                            │    │       │
│  │ 2. return (True, q_sample, response_data)  │────┤       │
│  └────────────────────────────────────────────┘    │       │
│                                                     │       │
│  ┌────────────────────────────────────────────┐    │       │
│  │ Worker Thread N                            │    │       │
│  │                                            │    │       │
│  │ 1. process_single_async(q_sample)          │    │       │
│  │    └─> response_data = _process_api_single()│   │       │
│  │        └─> HTTP Request to API Server      │    │       │
│  │        └─> return _process_single_response()│   │       │
│  │            └─> return {...} ✅              │    │       │
│  │                                            │    │       │
│  │ 2. return (True, q_sample, response_data)  │────┘       │
│  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   MLPerf LoadGen Library     │
              │                              │
              │  Receives callbacks from     │
              │  SINGLE THREAD (main thread) │
              │                              │
              │  ✅ No deadlock!              │
              │  ✅ Thread-safe!              │
              └──────────────────────────────┘

BENEFITS:
✅ Only main thread calls lg.QuerySamplesComplete()
✅ No concurrent LoadGen callbacks
✅ No race conditions on LoadGen state
✅ Predictable, single-threaded callback context
✅ Workers return data (functional pattern)
```

## Code Path Comparison

### BEFORE: Side-Effect Based (Broken)

```python
# Main Thread
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_single_async, qs) for qs in query_samples]

    for future in as_completed(futures):
        success, query_id = future.result()  # ← Only status
        # Can't do anything here, LoadGen already called!
        processed_samples += 1

# Worker Thread (concurrent with others)
def process_single_async(q_sample):
    self._process_api_single(q_sample, ...)  # ← Side effects!
    return (True, q_sample.id)

def _process_api_single(q_sample, ...):
    # Make HTTP request
    response = requests.post(...)

    # Process and CALL LOADGEN (PROBLEM!)
    self._process_single_response(...)  # ← No return value
    # Returns None

def _process_single_response(query_id, ...):
    # Create numpy array
    token_array = np.ascontiguousarray(...)
    self.response_arrays[query_id] = token_array  # ← Race condition!

    # Create LoadGen response
    response_array = [lg.QuerySampleResponse(...)]

    # PROBLEM: Called from worker thread!
    lg.QuerySamplesComplete(response_array)  # ❌ CONCURRENT CALLS!
    # Returns None
```

### AFTER: Data-Return Based (Fixed)

```python
# Main Thread
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_single_async, qs) for qs in query_samples]

    for future in as_completed(futures):
        success, q_sample_result, response_data = future.result()  # ← Full data!

        if success and response_data:
            # Extract data
            query_id = response_data['query_id']
            output_data_ptr = response_data['output_data_ptr']
            output_data_size = response_data['output_data_size']
            n_tokens = response_data['n_tokens']

            # Create LoadGen response
            response_array = [lg.QuerySampleResponse(
                query_id, output_data_ptr, output_data_size, n_tokens
            )]

            # SOLUTION: Call from main thread only!
            lg.QuerySamplesComplete(response_array)  # ✅ SINGLE-THREADED!

        processed_samples += 1

# Worker Thread (concurrent with others)
def process_single_async(q_sample):
    response_data = self._process_api_single(q_sample, ...)  # ← Returns data!
    return (True, q_sample, response_data)  # ← Pass data to main thread

def _process_api_single(q_sample, ...):
    # Make HTTP request
    response = requests.post(...)

    # Process and RETURN DATA (no side effects!)
    return self._process_single_response(...)  # ← Returns dictionary

def _process_single_response(query_id, ...):
    # Create numpy array
    token_array = np.ascontiguousarray(...)
    self.response_arrays[query_id] = token_array  # ← Still stored (for GC)

    # Prepare LoadGen response data
    output_data_ptr = token_array.ctypes.data
    output_data_size = token_array.nbytes
    n_tokens = len(token_ids)

    # SOLUTION: Return data instead of calling LoadGen!
    return {
        'query_id': query_id,
        'output_data_ptr': output_data_ptr,
        'output_data_size': output_data_size,
        'n_tokens': n_tokens
    }  # ✅ RETURNS DATA!
```

## Function Call Stack Comparison

### BEFORE (Broken):

```
Main Thread:
  issue_queries()
    └─> ThreadPoolExecutor.submit(process_single_async) × N
    └─> as_completed(futures)
        └─> future.result() → (success, query_id)
        └─> [Nothing to do here, side effects already happened]

Worker Thread 1...N (CONCURRENT):
  process_single_async(q_sample)
    └─> _process_api_single(q_sample, ...)
        └─> requests.post(...)
        └─> _process_single_response(...)
            └─> np.ascontiguousarray(...)
            └─> self.response_arrays[query_id] = ...  ← Race!
            └─> lg.QuerySampleResponse(...)
            └─> lg.QuerySamplesComplete(...)  ❌ Concurrent!
    └─> return (True, q_sample.id)
```

### AFTER (Fixed):

```
Main Thread:
  issue_queries()
    └─> ThreadPoolExecutor.submit(process_single_async) × N
    └─> as_completed(futures)
        └─> future.result() → (success, q_sample, response_data)
        └─> Extract response_data dictionary fields
        └─> lg.QuerySampleResponse(...)
        └─> lg.QuerySamplesComplete(...)  ✅ Single-threaded!

Worker Thread 1...N (CONCURRENT):
  process_single_async(q_sample)
    └─> _process_api_single(q_sample, ...)
        └─> requests.post(...)
        └─> _process_single_response(...)
            └─> np.ascontiguousarray(...)
            └─> self.response_arrays[query_id] = ...  ✓ Still done
            └─> return {'query_id': ..., ...}  ✅ Returns data!
        └─> return response_data
    └─> return (True, q_sample, response_data)
```

## Timeline Diagram

### BEFORE (Hangs):

```
Time ──▶

Main Thread:    [Submit Tasks]───[Wait for Futures]─────────────[HUNG]
                                          │
Worker Thread 1:  │──[HTTP]──[Process]──[CallLoadGen]──┐
                                                        │
Worker Thread 2:     │──[HTTP]──[Process]──[CallLoadGen]──┤
                                                           ├──▶ DEADLOCK!
Worker Thread 3:        │──[HTTP]──[Process]──[CallLoadGen]─┤
                                                             │
LoadGen:                                              [Locked]──[Waiting]──[Hung]
```

### AFTER (Works):

```
Time ──▶

Main Thread:    [Submit Tasks]───[Wait]───[Receive Data]───[CallLoadGen]──[CallLoadGen]──[CallLoadGen]──[Done]
                                               ▲                 ▲              ▲              ▲
                                               │                 │              │              │
Worker Thread 1:  │──[HTTP]──[Process]──[Return Data]──────────┘              │              │
                                                                                │              │
Worker Thread 2:     │──[HTTP]──[Process]──[Return Data]──────────────────────┘              │
                                                                                               │
Worker Thread 3:        │──[HTTP]──[Process]──[Return Data]───────────────────────────────────┘

LoadGen:                                                      [Process]───[Process]───[Process]──[Done]
```

## Summary Table

| Aspect | BEFORE (Broken) | AFTER (Fixed) |
|--------|----------------|---------------|
| **Worker Thread Role** | Make HTTP request + Call LoadGen | Make HTTP request + Return data |
| **Main Thread Role** | Submit tasks + Wait | Submit tasks + Wait + Call LoadGen |
| **LoadGen Callback Context** | Multiple worker threads (concurrent) | Single main thread (sequential) |
| **Thread Safety** | ❌ Race conditions, deadlocks | ✅ Thread-safe |
| **Function Return Types** | `None` (side effects) | `Dict[str, Any]` (data) |
| **Data Flow** | Worker → LoadGen directly | Worker → Main → LoadGen |
| **Pattern** | Imperative (side effects) | Functional (return values) |
| **Matches offline_sut.py** | ❌ No | ✅ Yes |
| **Production Ready** | ❌ Hangs | ✅ Works |

## Key Insight

The fundamental difference is:

**BEFORE**: "Do the thing" (imperative, side effects)
- Worker threads **do** the LoadGen callback

**AFTER**: "Give me the data" (functional, return values)
- Worker threads **return** data for the LoadGen callback
- Main thread **does** the LoadGen callback

This is a classic concurrent programming pattern: separate data generation (workers) from coordination (main thread).

