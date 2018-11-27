package jsat.parallelization;

import java.util.concurrent.RejectedExecutionException;

public interface ParallelizationEngine {
	public void start();
	public void stop();
	public void pause();
	
	public boolean canPause();
	
	public void addTask(Runnable task) throws RejectedExecutionException;
	public boolean offerTask(Runnable task);
}
