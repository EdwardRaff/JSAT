
package jsat.utils;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 *
 * Provides a fake {@link ExecutorService} that immediatly runs any 
 * given runnable on the main thread. This will not use any given 
 * threads and will never create a thread. 
 * 
 * @author Edward Raff
 */
public class FakeExecutor implements ExecutorService
{

    @Override
    public void shutdown()
    {
        
    }

    @Override
    public List<Runnable> shutdownNow()
    {
       return null;
    }

    @Override
    public boolean isShutdown()
    {
        return false;
    }

    @Override
    public boolean isTerminated()
    {
        return false;
    }

    @Override
    public boolean awaitTermination(long l, TimeUnit tu) throws InterruptedException
    {
        return true;
    }

    @Override
    public <T> Future<T> submit(final Callable<T> clbl)
    {
        return new Future<T>() {

        @Override
            public boolean cancel(boolean bln)
            {
                return false;
            }

        @Override
            public boolean isCancelled()
            {
                return false;
            }

        @Override
            public boolean isDone()
            {
                return false;
            }

        @Override
            public T get() throws InterruptedException, ExecutionException
            {
                try
                {
                    return clbl.call();
                }
                catch (Exception ex)
                {
                    return null;
                }
            }

        @Override
            public T get(long l, TimeUnit tu) throws InterruptedException, ExecutionException, TimeoutException
            {
                return get();
            }
        };
    }

    @Override
    public <T> Future<T> submit(Runnable r, T t)
    {
        r.run();
        return null;
    }

    @Override
    public Future<?> submit(Runnable r)
    {
        r.run();
        return null;
    }

    @Override
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> clctn) throws InterruptedException
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public <T> List<Future<T>> invokeAll(Collection<? extends Callable<T>> clctn, long l, TimeUnit tu) throws InterruptedException
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public <T> T invokeAny(Collection<? extends Callable<T>> clctn) throws InterruptedException, ExecutionException
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public <T> T invokeAny(Collection<? extends Callable<T>> clctn, long l, TimeUnit tu) throws InterruptedException, ExecutionException, TimeoutException
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void execute(Runnable r)
    {
        r.run();
    }
    
}
