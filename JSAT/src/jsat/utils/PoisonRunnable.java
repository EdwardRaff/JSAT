
package jsat.utils;

import java.util.concurrent.BlockingQueue;

/**
 * A helper class for using the reader / writer model to implement parallel algorithms. 
 * When using the a {@link BlockingQueue}, <tt>null</tt> is an invalid value. However,
 * if several threads have been started that are simply reading from the queue for 
 * jobs, we want to let the threads know when to stop. Polling can not be used since 
 * the {@link BlockingQueue#take() } operations blocks until a value is added. If 
 * the queue is a queue of {@link Runnable} jobs, this class solves the problem. 
 * An <tt>instanceof</tt> check can be done on the returned runnable. If it is a poisoned
 * one, the worker knows to stop consuming and terminate.  
 * 
 * @author Edward Raff
 */
public final class PoisonRunnable implements Runnable
{

    /**
     * This run method will throw an exception. 
     * @throws UnsupportedOperationException poisoned Runnables can not be run. 
     */
    @Override
    public void run()
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
