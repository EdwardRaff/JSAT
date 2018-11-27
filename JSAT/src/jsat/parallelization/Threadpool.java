package jsat.parallelization;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Threadpool implements ParallelizationEngine {
	private ThreadPoolExecutor pool;
	private BlockingQueue<Runnable> taskQueue;
	
	public Threadpool(int maxThreads) {
		taskQueue = new ArrayBlockingQueue<Runnable>(80);
		pool = new ThreadPoolExecutor(0, maxThreads, 1000, TimeUnit.MILLISECONDS, taskQueue);
	}

	@Override
	public void start() { }

	@Override
	public void stop() {
		pool.shutdown();
	}

	@Override
	public void pause() {
		throw new UnsupportedOperationException("The pause operation is not supported for the Threadpool");
	}

	@Override
	public boolean canPause() {
		return false;
	}

	@Override
	public void addTask(Runnable task) throws RejectedExecutionException {
		pool.execute(task);
	}

	@Override
	public boolean offerTask(Runnable task) {
		try {
			pool.execute(task);
			return true;
		} catch (RejectedExecutionException exception) {
			return false;
		}
	}

}
