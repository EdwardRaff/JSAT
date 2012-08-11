
package jsat.utils;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The RunnableConsumer is meant to be used in conjunction with 
 * {@link PoisonRunnable} and an {@link ExecutorService} to implement a 
 * consumer / produce model. It will consume runnables from a queue and 
 * immediately call its run method. Termination occurs when a 
 * {@link PoisonRunnable} is encountered, after it's run method is called.  
 * 
 * @author Edward Raff
 */
public class RunnableConsumer implements Runnable
{
    final private BlockingQueue<Runnable> jobQueue;

    /**
     * Creates a new runnable that will consume and run other runnables. 
     * @param jobQueue the queue from which to obtain runnable objects. 
     */
    public RunnableConsumer(BlockingQueue<Runnable> jobQueue)
    {
        this.jobQueue = jobQueue;
    }
    
    @Override
    public void run()
    {
        while(true)
        {
            try
            {
                Runnable toRun = jobQueue.take();

                toRun.run();

                if(toRun instanceof PoisonRunnable)
                    return;
            }
            catch (InterruptedException ex)
            {
                Logger.getLogger(RunnableConsumer.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
}
