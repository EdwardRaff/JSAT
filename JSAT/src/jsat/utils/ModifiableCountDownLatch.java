
package jsat.utils;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Edward Raff
 */
public class ModifiableCountDownLatch
{
    private Semaphore awaitSemaphore;
    AtomicInteger count;

    public ModifiableCountDownLatch(int count)
    {
        this.count = new AtomicInteger(count);
        awaitSemaphore = new Semaphore(0);
    }
    
    public void await() 
    {
        try
        {
            awaitSemaphore.acquire();
        }
        catch (InterruptedException ex)
        {
            System.err.println("AHHH What happened?");
        }
    }
    
    public void countDown()
    {
        if(count.decrementAndGet() == 0)
            awaitSemaphore.release();
    }
    
    public void countUp()
    {
        count.addAndGet(1);
    }
}
