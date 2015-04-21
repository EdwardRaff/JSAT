
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * The index table provides a way of accessing the sorted view of an array or list,
 * without ever sorting the elements of said list. Given an array of elements, the 
 * index table creates an array of index values, and sorts the indices based on 
 * the values they point to. The IndexTable can then be used to find the index 
 * of the i'th sorted element in the array. <br>
 * <br>
 * The IndexTable can be sorted multiple times by calling the 
 * {@link #sort(java.util.List, java.util.Comparator) } methods. This can be 
 * called on inputs of varying size, and the internal order will be expanded 
 * when necessary. 
 * 
 * @author Edward Raff
 */
public class IndexTable implements Serializable
{

	private static final long serialVersionUID = -1917765351445664286L;
	static private final Comparator defaultComp = new Comparator()        
    {

        @Override
        public int compare(Object o1, Object o2)
        {
            Comparable co1 = (Comparable) o1;
            Comparable co2 = (Comparable) o2;
            return co1.compareTo(co2);
        }
    };
    
    /**
     * Obtains the reverse order comparator
     * @param <T> the data type
     * @param cmp the original comparator
     * @return the reverse order comparator
     */
    public static <T> Comparator<T> getReverse(final Comparator<T> cmp)
    {
        return new Comparator<T>() 
        {
            @Override
            public int compare(T o1, T o2)
            {
                return -cmp.compare(o1, o2);
            }
        };
    }
    
    /**
     * We use an array of Integer objects instead of integers because we need 
     * the arrays.sort function that accepts comparators. 
     */
    private IntList index;
    /**
     * The size of the previously sorted array or list
     */
    private int prevSize;
    
    /**
     * Creates a new index table of a specified size that is in linear order. 
     * @param size the size of the index table to create
     */
    public IndexTable(int size)
    {
        index = new IntList(size);
        ListUtils.addRange(index, 0, size, 1);
    }

    /**
     * Creates a new index table based on the given array. The array will not be altered. 
     * @param array the array to create an index table for. 
     */
    public IndexTable(double[] array)
    {
        this(DoubleList.unmodifiableView(array, array.length));
    }
    
    /**
     * Creates a new index table based on the given array. The array will not be altered. 
     * @param array the array to create an index table for
     */
    public <T extends Comparable<T>> IndexTable(T[] array)
    {
        index = new IntList(array.length);
        ListUtils.addRange(index, 0, array.length, 1);
        Collections.sort(index, new IndexViewCompG(array));
    }
    
    /**
     * Creates a new index table based on the given list. The list will not be altered. 
     * @param list the list to create an index table for
     */
    public <T extends Comparable<T>> IndexTable(List<T> list)
    {
        this(list, defaultComp);
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
        index = new IntList(list.size());
        ListUtils.addRange(index, 0, list.size(), 1);
        sort(list, comparator);
    }
    
    /**
     * Resets the index table so that the returned indices are in linear order, 
     * meaning the original input would be returned in its original order 
     * instead of sorted order. 
     */
    public void reset()
    {
        for(int i = 0; i < index.size(); i++)
            index.set(i, i);
    }
    
    /**
     * Reverse the current index order
     */
    public void reverse()
    {
        Collections.reverse(index);
    }
    
    /**
     * Adjusts this index table to contain the sorted index order for the given 
     * array
     * @param array the input to get sorted order of
     */
    public void sort(double[] array)
    {
        sort(DoubleList.unmodifiableView(array, array.length));
    }
    
    /**
     * Adjusts this index table to contain the reverse sorted index order for 
     * the given array
     * @param array the input to get sorted order of
     */
    public void sortR(double[] array)
    {
        sortR(DoubleList.unmodifiableView(array, array.length));
    }
    
    /**
     * Adjust this index table to contain the sorted index order for the given 
     * list
     * @param <T> the data type
     * @param list the list of objects
     */
    public <T extends Comparable<T>> void sort(List<T> list)
    {
        sort(list, defaultComp);
    }
    
    /**
     * Adjusts this index table to contain the reverse sorted index order for 
     * the given list
     * @param <T> the data type
     * @param list the list of objects
     */
    public <T extends Comparable<T>> void sortR(List<T> list)
    {
        sort(list, getReverse(defaultComp));
    }
    
    
    /**
     * Sets up the index table based on the given list of the same size and 
     * comparator. 
     * 
     * @param <T> the type in use
     * @param list the list of points to obtain a sorted IndexTable for
     * @param cmp the comparator to determined the sorted order
     */
    public <T> void sort(List<T> list, Comparator<T> cmp)
    {
        if(index.size() < list.size())
            for(int i = index.size(); i < list.size(); i++ )
                index.add(i);
        if(list.size() == index.size())
            Collections.sort(index, new IndexViewCompList(list, cmp));
        else
        {
            Collections.sort(index);//so [0, list.size) is at the front
            Collections.sort(index.subList(0, list.size()), new IndexViewCompList(list, cmp));
        }
        prevSize = list.size();
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
        Collections.swap(index, i, j);
    }
    
    /**
     * Given the index <tt>i</tt> into what would be the sorted array, the index in the unsorted original array is returned. <br>
     * If the original array was a double array, <i>double[] vals</i>, then the sorted order can be printed with <br>
     * <pre><code>
     * for(int i = 0; i &lt; indexTable.{@link #length() length}(); i++)
     *     System.out.println(vals[indexTable.get(i)]);
     * </code></pre>
     * @param i the index of the i'th sorted value
     * @return the index in the original list that would be in the i'th position
     */
    public int index(int i)
    {
        if(i >= prevSize || i < 0)
            throw new IndexOutOfBoundsException("The size of the previously sorted array/list is " + prevSize + " so index " + i + " is not valid");
        return index.get(i);
    }
    
    /**
     * The length of the previous array that was sorted
     * @return the length of the original array 
     */
    public int length()
    {
        return prevSize;
    }
    
    /**
     * Applies this index table to the specified target, putting {@code target} 
     * into the same ordering as this IndexTable. 
     * 
     * @throws RuntimeException if the length of the target array is not the same as the index table
     */
    public void apply(double[] target)
    {
        //use DoubleList view b/d we are only using set ops, so we wont run into an issue of re-allocating the array
        apply(DoubleList.view(target, target.length), new DoubleList(target.length));
    }
    
    /**
     * Applies this index table to the specified target, putting {@code target} 
     * into the same ordering as this IndexTable. 
     * 
     * @throws RuntimeException if the length of the target List is not the same as the index table
     */
    public void apply(List target)
    {
        apply(target, new ArrayList(target.size()));
    }
    
    /**
     * Applies this index table to the specified target, putting {@code target} 
     * into the same ordering as this IndexTable. It will use the provided 
     * {@code tmp} space to store the original values in target in the same 
     * ordering. It will be modified, and may be expanded using the {@link 
     * List#add(java.lang.Object) add} method if it does not contain sufficient 
     * space. Extra size in the tmp list will be ignored. After this method is 
     * called, {@code tmp} will contain the same ordering that was in 
     * {@code target} <br>
     * <br>
     * This method is provided as a means to reducing memory use when multiple 
     * lists need to be sorted. 
     * 
     * @param target the list to sort, that should be the same size as the 
     * previously sorted list. 
     * @param tmp the temp list that may be of any size
     */
    public void apply(List target, List tmp)
    {
        if (target.size() != length())
            throw new RuntimeException("target array does not have the same length as the index table");
        //fill tmp with the original ordering or target, adding when needed
        for (int i = 0; i < target.size(); i++)
            if (i >= tmp.size())
                tmp.add(target.get(i));
            else
                tmp.set(i, target.get(i));
        //place back into target from tmp to get sorted order
        for(int i = 0; i < target.size(); i++)
            target.set(i, tmp.get(index(i)));
    }
    
}
