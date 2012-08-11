
package jsat.utils;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A helper class for using the reader / writer model to implement parallel algorithms. 
 * When using the a {@link BlockingQueue}, <tt>null</tt> is an invalid value. However,
 * if several threads have been started that are simply reading from the queue for 
 * jobs, we want to let the threads know when to stop. Polling can not be used since 
 * the {@link BlockingQueue#take() } operations blocks until a value is added. If 
 * the queue is a queue of {@link Runnable} jobs, this class solves the problem. 
 * An <tt>instanceof</tt> check can be done on the returned runnable. If it is a poisoned
 * one, the worker knows to stop consuming and terminate.  
 * <br><br>
 * By default, the {@link #run() } method does not perform any action. The 
 * poison runnable can be used to perform ending clean up work, such as posting 
 * to a semaphore, by giving the work job to the 
 * {@link #PoisonRunnable(java.lang.Runnable) } constructor. The runnable given 
 * will be run when the Poison runnable's run method is called. 
 * 
 * @author Edward Raff
 */
public final class PoisonRunnable implements Runnable
{
    private Runnable lastStep;
    private CountDownLatch latch;
    private CyclicBarrier barrier;

    /**
     * Creates a new PoisonRunnable that will run the given runnable when it is called. 
     * @param lastStep the runnable to call when this runnable is finally called. 
     */
    public PoisonRunnable(Runnable lastStep)
    {
        this.lastStep = lastStep;
    }

    /**
     * Creates a new PoisonRunnable that will call the 
     * {@link CountDownLatch#countDown() } method on the given latch when its 
     * run method is finally called. 
     * 
     * @param latch the latch to decrement
     */
    public PoisonRunnable(CountDownLatch latch)
    {
        this.latch = latch;
    }

    /**
     * Creates a new PoisonRunnable that will call the 
     * {@link CyclicBarrier#await() } method of the given barrier when its run 
     * method is finally called. 
     * @param barrier the barrier to wait on. 
     */
    public PoisonRunnable(CyclicBarrier barrier)
    {
        this.barrier = barrier;
    }
    
    /**
     * Creates a new PoisonRunnable that will do nothing when its run method is called. 
     */
    public PoisonRunnable()
    {
        
    }
    
    @Override
    public void run()
    {
        try
        {
            if(lastStep != null)
                lastStep.run();
            if(latch != null)
                latch.countDown();
            if(barrier != null)
                barrier.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(PoisonRunnable.class.getName()).log(Level.SEVERE, null, ex);
        }
        catch (BrokenBarrierException ex)
        {
            Logger.getLogger(PoisonRunnable.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
