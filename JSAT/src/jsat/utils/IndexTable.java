
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * The index table provides a way of accessing the sorted view of an array or list,
 * without ever sorting the elements of said list. Given an array of elements, the 
 * index table creates an array of index values, and sorts the indices based on 
 * the values they point to. The IndexTable can then be used to find the index 
 * of the i'th sorted element in the array. 
 * 
 * @author Edward Raff
 */
public class IndexTable implements Serializable
{
    /**
     * We use an array of Integer objects instead of integers because we need the arrays.sort function that accepts comparators. 
     */
    private Integer[] index;

    /**
     * Creates a new index table based on the given array. The array will not be altered. 
     * @param array the array to create an index table for. 
     */
    public IndexTable(double[] array)
    {
        index = new Integer[array.length];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompD(array));
    }
    
    /**
     * Creates a new index table based on the given array. The array will not be altered. 
     * @param array the array to create an index table for
     */
    public <T extends Comparable<T>> IndexTable(T[] array)
    {
        index = new Integer[array.length];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompG(array));
    }
    
    /**
     * Creates a new index table based on the given list. The list will not be altered. 
     * @param list the list to create an index table for
     */
    public <T extends Comparable<T>> IndexTable(List<T> list)
    {
        this(list, new Comparator<T>() {

            @Override
            public int compare(T o1, T o2)
            {
                return o1.compareTo(o2);
            }
        });
    }
    
    /**
     * Creates a new index table based on the given list and comparator. The 
     * list will not be altered. 
     * 
     * @param list the list of points to obtain a sorted IndexTable for
     * @param comparator the comparator to determined the sorted order
     */
    public <T> IndexTable(List<T> list, Comparator<T> comparator)
    {
        index = new Integer[list.size()];
        for(int i = 0; i < index.length; i++)
            index[i] = i;
        Arrays.sort(index, new IndexViewCompList(list, comparator));
    }
    
    private class IndexViewCompD implements Comparator<Integer> 
    {
        double[] base;

        public IndexViewCompD(double[] base)
        {
            this.base = base;
        }
        
        public int compare(Integer t, Integer t1)
        {
            return Double.compare(base[t], base[t1]);
        }        
    }
    
    private class IndexViewCompG<T extends Comparable<T>> implements Comparator<Integer> 
    {
        T[] base;

        public IndexViewCompG(T[] base)
        {
            this.base = base;
        }

        @Override
        public int compare(Integer t, Integer t1)
        {
            return base[t].compareTo(base[t1]);
        }        
    }
    
    private class IndexViewCompList<T> implements Comparator<Integer> 
    {
        final List<T> base;
        final Comparator<T> comparator;
        
        public IndexViewCompList(List<T> base, Comparator<T> comparator)
        {
            this.base = base;
            this.comparator = comparator;
        }

        @Override
        public int compare(Integer t, Integer t1)
        {
            return comparator.compare(base.get(t), base.get(t1));
        }        
    }
    
    /**
     * Swaps the given indices in the index table. 
     * @param i the second index to swap 
     * @param j the first index to swap
     */
    public void swap(int i, int j)
    {
        int tmp = index[i];
        index[i] = index[j];
        index[j] = tmp;
    }
    
    /**
     * Given the index <tt>i</tt> into what would be the sorted array, the index in the unsorted original array is returned. <br>
     * If the original array was a double array, <i>double[] vals</i>, then the sorted order can be printed with <br>
     * <code><pre>
     * for(int i = 0; i &lt; indexTable.{@link #length() length}(); i++)
     *     System.out.println(vals[indexTable.get(i)]);
     * </pre></code>
     * @param i the index of the i'th sorted value
     * @return the index in the original list that would be in the i'th position
     */
    public int index(int i)
    {
        return index[i];
    }
    
    /**
     * The length of the original array that was sorted
     * @return the length of the original array 
     */
    public int length()
    {
        return index.length;
    }
    
    /**
     * Applies this index table to the specified target. The application will unsorted the index
     * table, returning it to a state of representing the un ordered list. 
     * @throws RuntimeException if the length of the target array is not the same as the index table
     */
    public void apply(double[] target)
    {
        if(target.length != length())
            throw new RuntimeException("target array does not have the same length as the index table");
        for(int i = 0; i < target.length; i++)
        {
            double tmp = target[i];
            target[i] = target[index(i)];
            target[index(i)] = tmp;
            swap(i, index(i));
        }
    }
    
    /**
     * Applies this index table to the specified target. The application will unsorted the index
     * table, returning it to a state of representing the un ordered list. 
     * @throws RuntimeException if the length of the target List is not the same as the index table
     */
    public void apply(List target)
    {
        if(target.size() != length())
            throw new RuntimeException("target array does not have the same length as the index table");
        for(int i = 0; i < target.size(); i++)
        {
            ListUtils.swap(target, i, index(i));
            swap(i, index(i));
        }
    }
    
}
