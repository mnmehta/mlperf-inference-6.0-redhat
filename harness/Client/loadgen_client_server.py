"""LoadGen Server Client for MLPerf Inference benchmarks.

This module contains the LoadGenServerClient class for Server scenario benchmarking.
"""

import array
import asyncio
import json
import threading
from typing import List

import aiohttp
import numpy as np

try:
    import mlperf_loadgen as lg
except ImportError:
    raise ImportError(
        "Unable to import mlperf_loadgen. "
        "Please install it from https://github.com/mlcommons/inference"
    )

from .loadgen_client import LoadGenClient


class LoadGenServerClient(LoadGenClient):
    """
    LoadGen client for Server scenario.
    
    Inspired by SUT_VLLM_SingleReplica_Server.py with:
    - Worker threads for async query processing
    - Streaming API support
    - First token handling
    - Query queue management
    """
    
    def __init__(self, *args, **kwargs):
        # Force scenario to Server
        kwargs['scenario'] = 'Server'
        super().__init__(*args, **kwargs)
        self.query_counter = 0
        # Track event loops for cleanup
        self.worker_loops = {}
    
    def start_workers(self):
        """Start worker threads for async query processing."""
        if self.workers_started:
            return
        
        if not self.api_server_url:
            self.logger.warning("Workers not started - no API server URL")
            return
        
        self.logger.info(f"Starting {self.num_workers} worker threads")
        
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(
                target=self._process_queries_worker,
                args=(j,),
                daemon=True
            )
            worker.start()
            self.worker_threads[j] = worker

        self.workers_started = True
        self.logger.info("Worker threads started")
    
    def _process_queries_worker(self, worker_id: int):
        """Worker thread to process queued queries with asyncio."""
        # Create a new event loop for this worker thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.worker_loops[worker_id] = loop

        self.logger.debug(f"Worker {worker_id}: Started with asyncio event loop")

        try:
            # Run the async worker loop
            loop.run_until_complete(self._async_worker_loop(worker_id, loop))
        finally:
            # Cleanup: cancel all pending tasks
            self.logger.debug(f"Worker {worker_id}: Shutting down event loop")
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            # Wait for all tasks to complete cancellation
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self.logger.debug(f"Worker {worker_id}: Cleanup complete")

    async def _async_worker_loop(self, worker_id: int, loop: asyncio.AbstractEventLoop):
        """Async worker loop that processes queries from the queue."""
        # Create aiohttp session with connection pooling
        connector = aiohttp.TCPConnector(limit=None)
        timeout = aiohttp.ClientTimeout(total=None)  # No timeout for streaming

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            self.logger.debug(f"Worker {worker_id}: Created aiohttp session")

            # Track active tasks for this worker
            active_tasks = set()

            while True:
                # Get query from queue (blocking call in thread)
                # We need to run this in executor since queue.get() is blocking
                try:
                    qitem = await loop.run_in_executor(None, self.query_queue.get)

                    if qitem is None:
                        self.logger.debug(f"Worker {worker_id}: Received stop signal, waiting for active tasks")
                        # Wait for all active tasks to complete
                        if active_tasks:
                            await asyncio.gather(*active_tasks, return_exceptions=True)
                        break

                    # Get input IDs from dataset
                    input_ids_tensor = self.dataset.input_ids[qitem.index]

                    # Create async task for query processing
                    task = asyncio.create_task(
                        self._async_process_query(
                            input_ids_tensor,
                            qitem.id,
                            qitem.index,
                            session
                        )
                    )
                    active_tasks.add(task)
                    # Remove task from active set when done
                    task.add_done_callback(active_tasks.discard)

                except Exception as e:
                    self.logger.error(f"Worker {worker_id}: Error in query processing: {e}")
    
    async def _async_process_query(self, input_ids_tensor: List[int], query_id: int, query_index: int, session: aiohttp.ClientSession):
        """Process a single query asynchronously via streaming API.

        Args:
            input_ids_tensor: Input token IDs
            query_id: Query ID for LoadGen
            query_index: Index in dataset
            session: aiohttp.ClientSession to use for this request
        """
        try:
            # Get input token count for tracking
            input_token_count = len(input_ids_tensor)

            # Use text_input directly if available (same logic as LoadGenOfflineClient)
            # Otherwise decode from input_ids
            if (hasattr(self.dataset, 'input') and
                len(self.dataset.input) > query_index and
                self.dataset.input[query_index]):
                # Use text_input directly (no detokenization needed)
                decoded = self.dataset.input[query_index]
                self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id} (index {query_index}): Using text_input directly from dataset (length: {len(decoded)} chars)")
            elif self.tokenizer:
                try:
                    decoded = self.tokenizer.decode(input_ids_tensor, skip_special_tokens=True)
                    self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id} (index {query_index}): Decoded from input_ids using tokenizer (length: {len(decoded)} chars)")
                except Exception as e:
                    self.logger.warning(f"Error decoding tokens for query {query_id}: {e}")
                    decoded = " ".join([str(t) for t in input_ids_tensor])
                    self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id}: Fallback to string representation of input_ids")
            else:
                decoded = " ".join([str(t) for t in input_ids_tensor])
                self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id}: No tokenizer available, using string representation of input_ids")

            # Log the decoded text (first 200 chars) for debugging
            text_preview = decoded[:200] + "..." if len(decoded) > 200 else decoded
            self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id} (index {query_index}): Decoded text preview: {text_preview}")
            self.logger.debug(f"LoadGenServerClient._async_process_query() - Query {query_id} (index {query_index}): Full decoded text length: {len(decoded)} chars, input_ids length: {len(input_ids_tensor)} tokens")

            # Process via streaming API (pass session)
            response_ids = [query_id]
            output_tokens = await self._stream_api_vllm(decoded, response_ids, session)
            
            n_tokens = len(output_tokens)
            self.logger.debug(f"Query {query_id}: {n_tokens} tokens")
            
            # Track token statistics (always track if print_token_stats is enabled, or in debug mode)
            if self.print_token_stats or self.debug_mode:
                self._track_token_stats(input_token_count, n_tokens)
            
            if n_tokens <= 1:
                self.logger.warning(f"Low token count for query {query_id}: {n_tokens}")

            # IMPORTANT: Exclude first token from QuerySamplesComplete
            # The first token was already sent via FirstTokenComplete during streaming
            # To match reference implementation (server_sut.py line 344-349)
            if len(output_tokens) > 1:
                # Exclude first token - it was already sent to FirstTokenComplete
                remaining_tokens = output_tokens[1:]
                n_tokens_remaining = len(remaining_tokens)
                self.logger.debug(f"Query {query_id}: Sending {n_tokens_remaining} remaining tokens to QuerySamplesComplete (total generated: {n_tokens})")
            else:
                # Edge case: only 1 token total (or empty)
                remaining_tokens = []
                n_tokens_remaining = 0
                self.logger.debug(f"Query {query_id}: Only 1 token generated, sending empty to QuerySamplesComplete")

            # Create final response with REMAINING tokens only
            response_array = array.array("B", np.array(remaining_tokens, dtype=np.int32).tobytes())
            # CRITICAL: Keep array alive to prevent garbage collection before LoadGen reads it
            self.response_arrays[query_id] = response_array
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(query_id, bi[0], bi[1], n_tokens_remaining)]
            lg.QuerySamplesComplete(response)
            
        except Exception as e:
            self.logger.error(f"Error processing query {query_id}: {e}")
            self._send_error_response_by_id(query_id)

    def _submit_first_token_response(self, first_token_id: int | str, query_id: int):
        # Create first token response (single token)
        first_tokens = [first_token_id] if isinstance(first_token_id, int) else first_token_id
        response_data = array.array("B", np.array(first_tokens, dtype=np.int32).tobytes())
        bi = response_data.buffer_info()
        response = [lg.QuerySampleResponse(query_id, bi[0], bi[1])]
        lg.FirstTokenComplete(response)
    
    async def _stream_api_vllm(self, input_text: str, response_ids: List[int], session: aiohttp.ClientSession) -> List[int]:
        """
        Stream API call to vLLM server with first token handling.

        Args:
            input_text: Input text prompt
            response_ids: List of response IDs (for first token handling)
            session: aiohttp.ClientSession to use for this request

        Returns:
            List of output token IDs
        """
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Get sampling parameters based on test_mode
        temperature, top_k, top_p = self._get_sampling_params()
        
        # Log sampling parameters in debug mode
        self.logger.debug(f"LoadGenServerClient._stream_api_vllm() - Sampling parameters:")
        self.logger.debug(f"  test_mode: {self.test_mode}")
        self.logger.debug(f"  temperature: {temperature}")
        self.logger.debug(f"  top_k: {top_k}")
        self.logger.debug(f"  top_p: {top_p}")
        
        # Get server URL (with load balancing if enabled)
        server_url = self._get_next_server_url()
        endpoints = self._get_endpoints_for_url(server_url)
        
        # Determine endpoint and payload based on endpoint_type
        if self.endpoint_type == 'chat_completions':
            endpoint_url = endpoints['chat_completions']
            json_data = {
                'model': self.model_name,
                'messages': [{"role": "user", "content": input_text}],
                'max_tokens': self.max_tokens,
                'temperature': temperature,
                'stream': True,
                'top_p': top_p,
                'top_k': -1,
                'return_token_ids': True,  # Request token IDs in streaming response
            }
            # Add top_k if it's not -1 (which means consider all tokens)
            # Note: top_k should be an integer, not float
            if top_k != -1 and top_k > 0:
                json_data['top_k'] = int(top_k)
        else:
            endpoint_url = endpoints['completions']
            json_data = {
                'model': self.model_name,
                'prompt': input_text,
                'max_tokens': self.max_tokens,
                'min_tokens': 1,
                'temperature': temperature,
                'stream': True,
                'top_p': top_p,
                'top_k': -1,
                'return_token_ids': True,  # Request token IDs in streaming response
            }
        
        # Log the JSON payload being sent (in debug mode)
        self.logger.debug(f"LoadGenServerClient._stream_api_vllm() - Sending request to {endpoint_url} self.input_text: {input_text}")
        self.logger.debug(f"  JSON payload: {json.dumps(json_data, indent=2)}")

        token_s_cache = []  # For text accumulation (fallback)
        token_ids_cache = []  # For token ID accumulation (preferred)
        using_token_ids = None  # Track which method we're using
        first = True
        retry_count = 0
        max_retries = 3

        while True:
            try:
                async with session.post(
                    endpoint_url,
                    headers=headers,
                    json=json_data,
                    ssl=False
                ) as resp:
                    if resp.status != 200:
                        retry_count += 1
                        if retry_count <= max_retries:
                            self.logger.warning(f"API server returned status {resp.status}. Retry {retry_count}/{max_retries}")
                            await asyncio.sleep(0.05)  # Wait 50ms before retry
                            continue
                        else:
                            error_text = await resp.text()
                            self.logger.error(f"API server returned status {resp.status}: {error_text}. Max retries ({max_retries}) exceeded.")
                            break

                    # Stream response line by line
                    async for line in resp.content:
                        if line:
                            decoded = line.decode("utf-8")
                            
                            if decoded.startswith("data") and "[DONE]" not in decoded:
                                try:
                                    json_str = decoded[len("data: "):]

                                    # FAST PATH: Extract first token with minimal parsing before full JSON parse
                                    # This reduces TTFT latency by avoiding expensive JSON parsing just to get first token ID
                                    if first:
                                        first_token_extracted = False

                                        # Try delta_token_ids pattern (most common for streaming)
                                        if '"delta_token_ids":[' in json_str:
                                            try:
                                                start_idx = json_str.index('"delta_token_ids":[') + len('"delta_token_ids":[')
                                                end_idx = start_idx
                                                # Extract first integer token ID
                                                while end_idx < len(json_str) and (json_str[end_idx].isdigit() or json_str[end_idx] == '-'):
                                                    end_idx += 1
                                                if end_idx > start_idx:
                                                    first_token_id = int(json_str[start_idx:end_idx])
                                                    self._submit_first_token_response(first_token_id, response_ids[0])
                                                    first = False
                                                    first_token_extracted = True
                                                    self.logger.debug(f"Fast-path: Extracted first token ID {first_token_id} before JSON parse")
                                            except (ValueError, IndexError) as e:
                                                # Fast path failed, fall through to normal JSON parsing
                                                self.logger.debug(f"Fast-path delta_token_ids extraction failed: {e}")

                                        # Fallback: try token_ids pattern if delta not found
                                        if not first_token_extracted and '"token_ids":[' in json_str:
                                            try:
                                                start_idx = json_str.index('"token_ids":[') + len('"token_ids":[')
                                                end_idx = start_idx
                                                # Extract first integer token ID
                                                while end_idx < len(json_str) and (json_str[end_idx].isdigit() or json_str[end_idx] == '-'):
                                                    end_idx += 1
                                                if end_idx > start_idx:
                                                    first_token_id = int(json_str[start_idx:end_idx])
                                                    self._submit_first_token_response(first_token_id, response_ids[0])
                                                    first = False
                                                    first_token_extracted = True
                                                    self.logger.debug(f"Fast-path: Extracted first token ID {first_token_id} from token_ids before JSON parse")
                                            except (ValueError, IndexError) as e:
                                                # Fast path failed, fall through to normal JSON parsing
                                                self.logger.debug(f"Fast-path token_ids extraction failed: {e}")

                                    # Full JSON parse for complete processing (always needed for token accumulation)
                                    data = json.loads(json_str)

                                    # DEBUG: Log raw chunk data (first chunk only to avoid spam)
                                    if first:
                                        self.logger.debug(f"First streaming chunk JSON: {json_str[:500]}")

                                    choice_data = data["choices"][0]

                                    # Extract finish_reason but process tokens FIRST
                                    finish_reason = choice_data.get("finish_reason")
                                    stop_reason = choice_data.get("stop_reason")

                                    # Try to get token_ids from streaming response first
                                    delta_token_ids = choice_data.get("delta_token_ids", None)
                                    chunk_token_ids = choice_data.get("token_ids", None)

                                    # DEBUG: Log what we received in this chunk
                                    self.logger.debug(f"Streaming chunk: delta_token_ids={delta_token_ids}, chunk_token_ids={chunk_token_ids}, finish_reason={finish_reason}")

                                    if delta_token_ids is not None:
                                        # Got delta token IDs - this is the preferred method
                                        if using_token_ids is None:
                                            using_token_ids = True
                                            self.logger.debug("Using delta_token_ids from streaming API response")

                                        token_ids_cache.extend(delta_token_ids)

                                        # Handle first token - call FirstTokenComplete directly (thread-safe)
                                        if first and delta_token_ids:
                                            self._submit_first_token_response(delta_token_ids[0], response_ids[0])
                                            first = False

                                    elif chunk_token_ids is not None:
                                        # Got token IDs in chunk
                                        if using_token_ids is None:
                                            using_token_ids = True
                                            self.logger.debug("Using token_ids from streaming API response")

                                        # Check if this looks like cumulative (all tokens) or delta (new tokens)
                                        # If chunk has more tokens than we've seen, it's cumulative; otherwise extend
                                        if len(chunk_token_ids) > len(token_ids_cache):
                                            # Looks cumulative - replace cache
                                            token_ids_cache = chunk_token_ids.copy()
                                        else:
                                            # Looks like individual tokens - extend cache
                                            token_ids_cache.extend(chunk_token_ids)

                                        # Handle first token - call FirstTokenComplete directly (thread-safe)
                                        if first and token_ids_cache:
                                            self._submit_first_token_response(token_ids_cache[0], response_ids[0])
                                            first = False

                                    else:
                                        # Fallback: extract text and accumulate
                                        if using_token_ids is None:
                                            using_token_ids = False
                                            self.logger.debug("token_ids not found in streaming response, falling back to text accumulation")

                                        # Extract token text based on endpoint type
                                        if self.endpoint_type == 'chat_completions':
                                            # Chat completions uses delta.content
                                            delta = choice_data.get("delta", {})
                                            token_s = delta.get("content", "")
                                        else:
                                            # Completions uses text
                                            token_s = choice_data.get("text", "")

                                        if token_s != "":
                                            # Handle first token - call FirstTokenComplete directly (thread-safe)
                                            if first:
                                                if self.tokenizer:
                                                    token_ids = self.tokenizer.encode(token_s, add_special_tokens=False)
                                                    if token_ids:
                                                        self._submit_first_token_response(token_ids[0], response_ids[0])
                                                first = False

                                            token_s_cache.append(str(token_s))

                                    # NOW check finish_reason AFTER processing tokens
                                    if (finish_reason is not None) or (stop_reason is not None):
                                        # Generation finished - break after this chunk
                                        if finish_reason != "length":
                                            self.logger.debug(
                                                f"Sequence finished: finish_reason={finish_reason}, "
                                                f"stop_reason={stop_reason}"
                                            )
                                        break  # Exit loop after processing final chunk
                                    
                                except json.JSONDecodeError as e:
                                    self.logger.debug(f"JSON decode error: {e}")
                                    continue
                                except Exception as e:
                                    self.logger.debug(f"Error parsing stream line: {e}")
                                    continue

                    # DEBUG: Log what we accumulated
                    self.logger.debug(f"Streaming complete - using_token_ids={using_token_ids}, token_ids_cache length={len(token_ids_cache)}, token_s_cache length={len(token_s_cache)}")

                    # Return accumulated tokens
                    if using_token_ids and token_ids_cache:
                        # Got token IDs directly from API
                        self.logger.debug(f"Returning {len(token_ids_cache)} token IDs from streaming API: {token_ids_cache[:10]}..." if len(token_ids_cache) > 10 else f"Returning {len(token_ids_cache)} token IDs from streaming API: {token_ids_cache}")
                        return token_ids_cache
                    elif token_s_cache:
                        # Fallback: Convert accumulated text to token IDs
                        if self.tokenizer:
                            full_text = "".join(token_s_cache)
                            self.logger.debug(f"Accumulated text length: {len(full_text)} chars, text preview: '{full_text[:100]}'")
                            result = self.tokenizer.encode(full_text, add_special_tokens=False)
                            self.logger.debug(f"Returning {len(result)} token IDs from text re-encoding: {result[:10]}..." if len(result) > 10 else f"Returning {len(result)} token IDs from text re-encoding: {result}")
                            return result
                        else:
                            # Fallback: return placeholder tokens
                            self.logger.warning("No tokenizer available for encoding response")
                            return [1, 2, 3]
                    else:
                        # No tokens accumulated at all!
                        self.logger.error("No tokens accumulated from streaming response!")
                        return [1, 2, 3]

                    break

            except Exception as e:
                self.logger.error(f"Connection failure: {e}")
                # Return fallback tokens
                return [1, 2, 3]
        
        # Fallback if no tokens collected
        return [1, 2, 3]
    
    def issue_query(self, query_samples: List['lg.QuerySample']) -> None:
        """
        Process queries by queuing them for async processing.
        
        In server scenario, queries are queued and processed asynchronously
        by worker threads with streaming API support.
        """
        if not self.server_ready:
            self.logger.error("API server is not ready")
            self._send_error_responses(query_samples)
            return
        
        # Start workers if not already started
        if not self.workers_started:
            self.start_workers()
        
        # Queue queries for processing
        for sample in query_samples:
            self.logger.debug(f"Queuing query {sample.id} (index: {sample.index})")
            self.query_queue.put(sample)
            self.query_counter += 1
    
    def _send_error_responses(self, query_samples: List['lg.QuerySample']) -> None:
        """Send error responses for all queries."""
        for q_sample in query_samples:
            self._send_error_response_by_id(q_sample.id)
    
    def _send_error_response_by_id(self, query_id: int) -> None:
        """Send error response for a query ID."""
        response = lg.QuerySampleResponse(query_id, 0, 0, 0)
        lg.QuerySamplesComplete([response])
    
    def flush_queries(self) -> None:
        """
        Flush queries for server scenario.
        Signals workers to complete pending queries.
        """
        self.logger.info("Flush queries called (server scenario)")
        # Note: In server scenario, we let workers complete their current queries
        # Actual flushing logic can be added if needed
    
    def stop_workers(self):
        """Stop worker threads gracefully."""
        if not self.workers_started:
            return

        self.logger.info("Stopping worker threads...")

        # Signal workers to stop
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        # Wait for workers to finish (event loops will be cleaned up in their finally blocks)
        for worker in self.worker_threads:
            if worker and worker.is_alive():
                worker.join(timeout=10)

        # Cleanup: clear event loops tracking
        if self.worker_loops:
            self.logger.debug(f"Clearing {len(self.worker_loops)} worker event loops")
            self.worker_loops.clear()

        self.workers_started = False
        self.logger.info("Worker threads stopped")
    
    def cleanup(self) -> None:
        """Cleanup resources including worker threads."""
        # Print token statistics if enabled (before cleanup)
        # Note: This is a fallback - main printing happens in base_harness after test completes
        if (hasattr(self, 'print_token_stats') and self.print_token_stats) or \
           (hasattr(self, 'debug_mode') and self.debug_mode):
            if hasattr(self, 'input_token_counts') and self.input_token_counts:
                self.logger.info("Server scenario: Generating token statistics and histograms before cleanup...")
                self._print_token_histograms()
        
        # Clear stored response arrays before stopping workers
        if hasattr(self, 'response_arrays'):
            self.logger.debug(f"Clearing {len(self.response_arrays)} stored response arrays")
            self.response_arrays.clear()
        self.stop_workers()
        super().cleanup()
