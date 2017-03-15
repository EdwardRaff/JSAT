
package jsat.utils;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import jsat.utils.random.RandomUtil;

/**
 *
 * @author Edward Raff
 */
public class ListUtils
{

    private ListUtils()
    {
    }
    
    /**
     * This method takes a list and breaks it into <tt>count</tt> lists backed by the original 
     * list, with elements being equally spaced among the lists. The lists will be returned in
     * order of the consecutive values they represent in the source list.
     * <br><br><b>NOTE</b>: Because the implementation uses {@link List#subList(int, int) }, 
     * changes to the returned lists will be reflected in the source list. 
     * 
     * @param <T> the type contained in the list
     * @param source the source list that will be used to back the <tt>count</tt> lists
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
     * Returns a new unmodifiable view that is the merging of two lists
     * @param <T> the type the lists hold
     * @param left the left portion of the merged view
     * @param right the right portion of the merged view
     * @return a list view that contains bot the left and right lists
     */
    public static <T> List<T> mergedView(final List<T> left, final List<T> right)
    {
        List<T> merged = new AbstractList<T>() 
        {

            @Override
            public T get(int index)
            {
                if(index < left.size())
                    return left.get(index);
                else if(index-left.size() < right.size())
                    return right.get(index-left.size());
                else
                    throw new IndexOutOfBoundsException("List of lengt " + size() + " has no index " + index);
            }

            @Override
            public int size()
            {
                return left.size() + right.size();
            }
        };
        return merged;
    }
    
    /**
     * Swaps the values in the list at the given positions
     * @param list the list to perform the swap in
     * @param i the first position to swap
     * @param j the second position to swap
     */
    public static void swap(List list, int i, int j)
    {
        Object tmp = list.get(i);
        list.set(i, list.get(j));
        list.set(j, tmp);
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
    
    /**
     * Obtains a random sample without replacement from a source list and places
     * it in the destination list. This is done without modifying the source list. 
     * 
     * @param <T> the list content type involved
     * @param source the source of values to randomly sample from
     * @param dest the list to store the random samples in. The list does not 
     * need to be empty for the sampling to work correctly
     * @param samples the number of samples to select from the source
     * @param rand the source of randomness for the sampling
     * @throws IllegalArgumentException if the sample size is not positive or l
     * arger than the source population. 
     */
    public static <T> void randomSample(List<T> source, List<T> dest, int samples, Random rand)
    {
        randomSample((Collection<T>)source, dest, samples, rand);
    }
    
    /**
     * Obtains a random sample without replacement from a source collection and
     * places it in the destination collection. This is done without modifying 
     * the source collection. <br>
     * This random sampling is oblivious to failures to add to a collection that 
     * may occur, such as if the collection is a {@link Set}
     * 
     * @param <T> the list content type involved
     * @param source the source of values to randomly sample from
     * @param dest the collection to store the random samples in. It does 
     * not need to be empty for the sampling to work correctly
     * @param samples the number of samples to select from the source
     * @param rand the source of randomness for the sampling
     * @throws IllegalArgumentException if the sample size is not positive or l
     * arger than the source population. 
     */
    public static <T> void randomSample(Collection<T> source, Collection<T> dest, int samples, Random rand)
    {
        if(samples > source.size())
            throw new IllegalArgumentException("Can not obtain a number of samples larger than the source population");
        else if(samples <= 0)
            throw new IllegalArgumentException("Sample size must be positive");
        //Use samples to keep track of how many more samples we need
        int remainingPopulation = source.size();
        for(T member : source)
        {
            if(rand.nextInt(remainingPopulation) < samples)
            {
                dest.add(member);
                samples--;
            }
            remainingPopulation--;
        }
    }
    
    /**
     * Obtains a random sample without replacement from a source list and places
     * it in the destination list. This is done without modifying the source list. 
     * 
     * @param <T> the list content type involved
     * @param source the source of values to randomly sample from
     * @param dest the list to store the random samples in. The list does not 
     * need to be empty for the sampling to work correctly
     * @param samples the number of samples to select from the source
     * @throws IllegalArgumentException if the sample size is not positive or l
     * arger than the source population. 
     */
    public static <T> void randomSample(List<T> source, List<T> dest, int samples)
    {
        randomSample(source, dest, samples, RandomUtil.getRandom());
    }
}
