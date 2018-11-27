package jsat.parallelization;

public interface ParallelizationEngine {
	public void start();
	public void stop();
	public void pause();
	
	public boolean canPause();
	
	public void addTask(Runnable task);
	public boolean offerTask(Runnable task);
}
