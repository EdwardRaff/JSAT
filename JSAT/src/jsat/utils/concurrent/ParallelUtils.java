package jsat.utils.concurrent;
import static java.lang.Math.min;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BinaryOperator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import jsat.utils.FakeExecutor;
import jsat.utils.ListUtils;
import jsat.utils.SystemInfo;

/**
 *
 * @author Edward Raff
 */
public class ParallelUtils
{
    /**
     * This object provides a re-usable source of threads for use without having
     * to create a new thread pool. The cached executor service is used because
     * no threads are created until needed. If the pool is unused for a long
     * enough time, the threads will be destroyed. This avoids the user needing
     * to do anything. This pool is filled with daemon threads, and so will not
     * prevent program termination.
     */
    public static final ExecutorService CACHED_THREAD_POOL = Executors.newCachedThreadPool((Runnable r) ->
    {
        Thread t = Executors.defaultThreadFactory().newThread(r);
        t.setDaemon(true);
        return t;
    });
    
    /**
     * This helper method provides a convenient way to break up a computation
     * across <tt>N</tt> items into contiguous ranges that can be processed
     * independently in parallel.
     *
     * @param parallel a boolean indicating if the work should be done in
     * parallel. If false, it will run single-threaded. This is for code
     * convenience so that only one set of code is needed to handle both cases.
     * @param N the total number of items to process
     * @param lcr the runnable over a contiguous range 
     */
    public static void run(boolean parallel, int N, LoopChunkRunner lcr)
    {
        ExecutorService threadPool = Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        run(parallel, N, lcr, threadPool);
        threadPool.shutdownNow();
    }
    
    /**
     * This helper method provides a convenient way to break up a computation
     * across <tt>N</tt> items into contiguous ranges that can be processed
     * independently in parallel.
     *
     * @param parallel a boolean indicating if the work should be done in
     * parallel. If false, it will run single-threaded. This is for code
     * convenience so that only one set of code is needed to handle both cases.
     * @param N the total number of items to process. 
     * @param lcr the runnable over a contiguous range 
     * @param threadPool the source of threads for the computation 
     */
    public static void run(boolean parallel, int N, LoopChunkRunner lcr, ExecutorService threadPool)
    {
        if(!parallel)
        {
            lcr.run(0, N);
            return;
        }
        
        int cores_to_use = Math.min(SystemInfo.LogicalCores, N);
        final CountDownLatch latch = new CountDownLatch(cores_to_use);

        IntStream.range(0, cores_to_use).forEach(threadID ->
        {
            threadPool.submit(() ->
            {
                int start = ParallelUtils.getStartBlock(N, threadID, cores_to_use);
                int end = ParallelUtils.getEndBlock(N, threadID, cores_to_use);
                lcr.run(start, end);
                latch.countDown();
            });
        });

        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(ParallelUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    public static <T> T run(boolean parallel, int N, LoopChunkReducer<T> lcr, BinaryOperator<T> reducer, ExecutorService threadPool)
    {
        if(!parallel)
        {
            return lcr.run(0, N);
        }
        
        
        int cores_to_use = Math.min(SystemInfo.LogicalCores, N);
        final List<Future<T>> futures = new ArrayList<>(cores_to_use);
        

        IntStream.range(0, cores_to_use).forEach(threadID ->
        {
            futures.add(threadPool.submit(() ->
            {
                int start = ParallelUtils.getStartBlock(N, threadID, cores_to_use);
                int end = ParallelUtils.getEndBlock(N, threadID, cores_to_use);
                return lcr.run(start, end);
            }));
        });

        T cur = null;
        for(Future<T> ft : futures)
        {
            try
            {
                T chunk = ft.get();
                if(cur == null)
                    cur = chunk;
                else
                    cur = reducer.apply(cur, chunk);
            }
            catch (InterruptedException | ExecutionException ex)
            {
                Logger.getLogger(ParallelUtils.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return cur;
    }
    
    public static <T> T run(boolean parallel, int N, LoopChunkReducer<T> lcr, BinaryOperator<T> reducer)
    {
        ExecutorService threadPool = Executors.newWorkStealingPool(SystemInfo.LogicalCores);
        T toRet = run(parallel, N, lcr, reducer, threadPool);
        threadPool.shutdownNow();
        return toRet;
    }
    
    public static <T> T run(boolean parallel, int N, IndexReducer<T> ir, BinaryOperator<T> reducer)
    {
        if(!parallel)
        {
            T runner = ir.run(0);
            for(int i = 1; i < N; i++)
                runner = reducer.apply(runner, ir.run(i));
            return runner;
        }
        
        return range(N, parallel).mapToObj(j -> ir.run(j)).reduce(reducer).orElse(null);
    }
    
    public static void run(boolean parallel, int N, IndexRunnable ir)
    {
        ExecutorService threadPool = Executors.newWorkStealingPool(SystemInfo.LogicalCores);
        run(parallel, N, ir, threadPool);
        threadPool.shutdownNow();
    }
    
    /**
     * This helper method provides a convenient way to break up a computation
     * across <tt>N</tt> items into individual indices to be processed. This
     * method is meant for when the execution time of any given index is highly
     * variable, and so for load balancing purposes, should be treated as
     * individual jobs. If runtime is consistent, look at {@link #run(boolean, int, jsat.utils.concurrent.LoopChunkRunner, java.util.concurrent.ExecutorService)
     * }.
      
     *
     * @param parallel a boolean indicating if the work should be done in
     * parallel. If false, it will run single-threaded. This is for code
     * convenience so that only one set of code is needed to handle both cases.
     * @param N the total number of items to process. 
     * @param ir the runnable over a contiguous range 
     * @param threadPool the source of threads for the computation 
     */
    public static void run(boolean parallel, int N, IndexRunnable ir, ExecutorService threadPool)
    {
        if(!parallel)
        {
            for(int i = 0; i < N; i++)
                ir.run(i);
            return;
        }
        
        final CountDownLatch latch = new CountDownLatch(N);

        IntStream.range(0, N).forEach(threadID ->
        {
            threadPool.submit(() ->
            {
                ir.run(threadID);
                latch.countDown();
            });
        });

        try
        {
            latch.await();
        }
        catch (InterruptedException ex)
        {
            Logger.getLogger(ParallelUtils.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    public static ExecutorService getNewExecutor(boolean parallel)
    {
        if(parallel)
            return Executors.newFixedThreadPool(SystemInfo.LogicalCores);
        else
            return new FakeExecutor();
    }
    
    
    public static <T> Stream<T> streamP(Stream<T> source, boolean parallel)
    {
        if(parallel)
            return source.parallel();
        else
            return source;
    }
    
    public static IntStream streamP(IntStream source, boolean parallel)
    {
        if(parallel)
            return source.parallel();
        else
            return source;
    }
    
    public static DoubleStream streamP(DoubleStream source, boolean parallel)
    {
        if(parallel)
            return source.parallel();
        else
            return source;
    }
    
    public static IntStream range(int end,  boolean parallel)
    {
        return range(0, end, parallel);
    }
    
    public static IntStream range(int start, int end,  boolean parallel)
    {
        if(parallel)
        {
            /*
             * Why do this weirndes instead of call IntStream directly? 
             * IntStream has a habit of not returning a stream that actually 
             * executes in parallel when the range is small. That would make 
             * sense for most cases, but we probably are doing course 
             * parallelism into chunks. So this approach gurantees we get 
             * something that will actually run in parallel. 
             */
            return ListUtils.range(start, end).stream().parallel().mapToInt(i -> i);
        }
        else
            return IntStream.range(start, end);
    }
    
    /**
     * Gets the starting index (inclusive) for splitting up a list of items into
     * {@code P} evenly sized blocks. In the event that {@code N} is not evenly 
     * divisible by {@code P}, the size of ranges will differ by at most 1. 
     * @param N the number of items to split up
     * @param ID the block number in [0, {@code P})
     * @param P the number of blocks to break up the items into
     * @return the starting index (inclusive) of the blocks owned by the 
     * {@code ID}'th process. 
     */
    public static int getStartBlock(int N, int ID, int P)
    {
        int rem = N%P;
        int start = (N/P)*ID+min(rem, ID);
        return start;
    }
    
    /**
     * Gets the starting index (inclusive) for splitting up a list of items into
     * {@link SystemInfo#LogicalCores} evenly sized blocks. In the event that
     * {@code N} is not evenly divisible by {@link SystemInfo#LogicalCores}, the
     * size of ranges will differ by at most 1.
     *
     * @param N the number of items to split up
     * @param ID the block number in [0, {@link SystemInfo#LogicalCores})
     * @return the starting index (inclusive) of the blocks owned by the
     * {@code ID}'th process.
     */
    public static int getStartBlock(int N, int ID)
    {
        return getStartBlock(N, ID, SystemInfo.LogicalCores);
    }
    
    /**
     * Gets the ending index (exclusive) for splitting up a list of items into
     * {@code P} evenly sized blocks. In the event that {@code N} is not evenly 
     * divisible by {@code P}, the size of ranges will differ by at most 1. 
     * @param N the number of items to split up
     * @param ID the block number in [0, {@code P})
     * @param P the number of blocks to break up the items into
     * @return the ending index (exclusive) of the blocks owned by the 
     * {@code ID}'th process. 
     */
    public static int getEndBlock(int N, int ID, int P)
    {
        int rem = N%P;
        int start = (N/P)*(ID+1)+min(rem, ID+1);
        return start;
    }

    /**
     * Gets the ending index (exclusive) for splitting up a list of items into
     * {@link SystemInfo#LogicalCores} evenly sized blocks. In the event that
     * {@link SystemInfo#LogicalCores} is not evenly divisible by
     * {@link SystemInfo#LogicalCores}, the size of ranges will differ by at
     * most 1.
     *
     * @param N the number of items to split up
     * @param ID the block number in [0, {@link SystemInfo#LogicalCores})
     * @return the ending index (exclusive) of the blocks owned by the
     * {@code ID}'th process.
     */
    public static int getEndBlock(int N, int ID)
    {
        return getEndBlock(N, ID, SystemInfo.LogicalCores);
    }
}
