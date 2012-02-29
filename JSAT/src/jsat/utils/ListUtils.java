
package jsat.utils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Edward Raff
 */
public class ListUtils
{
    /**
     * This method takes a list and breaks it into <tt>chunks<tt> lists backed by the original 
     * list, with elements being equally spaced among the lists. The lists will be returned in
     * order of the consecutive values they represent in the source list.
     * <br><br><b>NOTE</b>: Because the implementation uses {@link List#subList(int, int) }, 
     * changes to the returned lists will be reflected in the source list. 
     * 
     * @param <T>
     * @param source the source list that will be used to back the chunked lists
     * @param count the number of lists to partition the source into. 
     * @return a lists of lists, each of with the same size with at most a difference of 1. 
     */
    public static <T> List<List<T>> splitList(List<T> source, int count)
    {
        if(count <= 0)
            throw new RuntimeException("Chunks must be greater then 0, not " + count);
        List<List<T>> chunks = new ArrayList<List<T>>(count);
        int baseSize = source.size() / count;
        int remainder = source.size() % count;
        int start = 0;
        
        
        for(int i = 0; i < count; i++)
        {
            int end = start+baseSize;
            if(remainder-- > 0)
                end++;
            chunks.add(source.subList(start, end));
            start = end;
        }
        
        return chunks;
    }
    
    /**
     * Collects all future values in a collection into a list, and returns said list. This method will block until all future objects are collected. 
     * @param <T> the type of future object
     * @param futures the collection of future objects
     * @return a list containing the object from the future. 
     * @throws ExecutionException 
     * @throws InterruptedException 
     */
    public static <T> List<T> collectFutures(Collection<Future<T>> futures) throws ExecutionException, InterruptedException
    {
        ArrayList<T> collected = new ArrayList<T>(futures.size());

        for (Future<T> future : futures)
                collected.add(future.get());

        return collected;
    }
    
    /**
     * Adds values into the given collection using integer in the specified range and step size. 
     * If the <tt>start</tt> value is greater or equal to the <tt>to</tt> value, nothing will 
     * be added to the collection. 
     * 
     * @param c the collection to add to 
     * @param start the first value to add, inclusive
     * @param to the last value to add, exclusive
     * @param step the step size. 
     * @throws RuntimeException if the step size is zero or negative.
     */
    public static void addRange(Collection<Integer> c, int start, int to, int step)
    {
        if(step <= 0)
            throw new RuntimeException("Would create an infinite loop");
        
        for(int i = start; i < to; i+= step)
            c.add(i);
    }
}
