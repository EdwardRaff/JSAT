
package jsat.utils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Provides a {@link CountDownLatch} that can have the number of counts increased as well as decreased. 
 * @author Edward Raff
 */
public class ModifiableCountDownLatch
{
    private Semaphore awaitSemaphore;
    private AtomicInteger count;

    public ModifiableCountDownLatch(int count)
    {
        this.count = new AtomicInteger(count);
        awaitSemaphore = new Semaphore(0);
    }
    
    /**
     * Waits until the count gets reduced to zero, and then all threads waiting will get to run. 
     * 
     * @throws InterruptedException 
     */
    public void await() throws InterruptedException
    {
        awaitSemaphore.acquire();
        awaitSemaphore.release();
    }
    
    /**
     * Decrements the counter. Allowing threads that have called {@link #await() } to run once the count reaches zero.
     */
    public void countDown()
    {
        if(count.get() == 0)
            return;
        else if( count.decrementAndGet() == 0)
            awaitSemaphore.release();
    }
    
    /**
     * Increments the count. Once the count has reached zero once, incrementing the count back above zero will have no affect. 
     */
    public void countUp()
    {
        count.addAndGet(1);
    }
}
