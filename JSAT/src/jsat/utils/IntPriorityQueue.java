
package jsat.utils;

import java.io.Serializable;
import java.util.*;

/**
 * This class represents a priority queue specifically designed to contain 
 * integer keys, and uses less memory then a {@link PriorityQueue} filled with 
 * integers. This queue can optionally support <i>log(n)</i> removal of key 
 * values at increased memory cost. 
 * @author Edward Raff
 */
public class IntPriorityQueue extends AbstractQueue<Integer> implements Serializable
{

	private static final long serialVersionUID = -310756323843109562L;
	public static final Comparator<Integer> naturalComparator = new Comparator<Integer>() {

        @Override
        public int compare(Integer o1, Integer o2)
        {
            return o1.compareTo(o2);
        }
    };
    
    /**
     * Sets the mode used for the priority queue.  
     */
    public static enum Mode
    {
        /**
         * Sets the priority queue to use the standard queue implementation with
         * the standard complexity for the following operations <br>
         * <br>
         * O(1) time for {@link #peek() } <br>
         * O(log n) time for {@link #poll() }, {@link #add(java.lang.Object) } <br>
         * O(n) time for {@link #remove(java.lang.Object)  } and {@link #contains(java.lang.Object) }<br>
         * And calling {@link Iterator#remove() } is O(log n), but the iteration is in no particular order<br>
         * 
         */
        STANDARD, 
        /**
         * Sets the priority queue to use a backing hashtable to map values to 
         * their position in the heap. This changes the complexity to <br>
         * <br>
         * O(1) time for {@link #peek() } and {@link #contains(java.lang.Object) } <br>
         * O(log n) time for {@link #poll() }, {@link #add(java.lang.Object) }, 
         * and {@link #remove(java.lang.Object)  } <br>
         * And calling {@link Iterator#remove() } is O(log n), but the iteration
         * is in no particular order<br>
         */
        HASH, 
        /**
         * Sets the priority queue to use a backing array that indexes into the 
         * heap. This provides the fastest version if fast 
         * {@link #remove(java.lang.Object) } and {@link #contains(java.lang.Object) }
         * are needed. However, it must create an array as large as the largest 
         * int value put into the queue, and negative values will not be accepted.
         * This method is best when the size of the queue is known to be bounded 
         * and all values are in the range [0, N].<br>
         * <br>
         * O(1) time for {@link #peek() } and {@link #contains(java.lang.Object) } <br>
         * O(log n) time for {@link #poll() }, {@link #add(java.lang.Object) }, 
         * and {@link #remove(java.lang.Object)  } <br>
         * And calling {@link Iterator#remove() } is O(log n), but the iteration
         * is in no particular order<br>
         */
        BOUNDED
    }
    
    private int[] heap;
    private int size;
    private Comparator<Integer> comparator;
    private final HashMap<Integer, Integer> valueIndexMap;
    private int[] valueIndexStore;
    private final Mode fastValueRemove;
    
    /**
     * Creates a new integer priority queue using {@link Mode#STANDARD}
     */
    public IntPriorityQueue()
    {
        this(8, naturalComparator);
    }
    
    /**
     * Creates a new integer priority queue using the specified comparison and {@link Mode#STANDARD}
     * @param initialSize the initial storage size of the queue
     * @param comparator the comparator to determine the order of elements in the queue
     */
    public IntPriorityQueue(int initialSize, Comparator<Integer> comparator)
    {
        this(initialSize, comparator, Mode.STANDARD);
    }
    
    /**
     * Creates a new integer priority queue
     * @param initialSize the initial storage size of the queue
     * @param fastValueRemove the mode that whether or not, and how, fast
     * arbitrary object removal from the queue will be done. 
     */
    public IntPriorityQueue(int initialSize, Mode fastValueRemove)
    {
        this(initialSize, naturalComparator, fastValueRemove);
    }
    
    /**
     * Creates a new integer priority queue
     * @param initialSize the initial storage size of the queue
     * @param comparator the comparator to determine the order of elements in the queue
     * @param fastValueRemove the mode that whether or not, and how, fast
     * arbitrary object removal from the queue will be done. 
     */
    public IntPriorityQueue( int initialSize, Comparator<Integer> comparator, Mode fastValueRemove)
    {
        this.heap = new int[initialSize];
        this.comparator = comparator;
        this.size = 0;
        this.fastValueRemove = fastValueRemove;
        valueIndexStore = null;
        if(fastValueRemove == Mode.HASH)
            valueIndexMap = new HashMap<Integer, Integer>(initialSize);
        else if(fastValueRemove == Mode.BOUNDED)
        {
            valueIndexStore = new int[initialSize];
            Arrays.fill(valueIndexStore, -1);
            valueIndexMap = null;
        }
        else
            valueIndexMap = null;
    }

    @Override
    public Iterator<Integer> iterator()
    {
        //Return a itterator that itterats the heap array in reverse. This is 
        //slower (against cache line), but allows for simple implementaiton of 
        //remove(). Becase removing an item is done by pulling the last element 
        //in the heap and then pushing down, all changes for removing node i 
        //will occur after index i
        return new Iterator<Integer>() 
        {
            int pos = size-1;
            boolean canRemove = false;
            @Override
            public boolean hasNext()
            {
                return pos >= 0;
            }

            @Override
            public Integer next()
            {
                canRemove = true;
                return heap[pos--];
            }

            @Override
            public void remove()
            {
                if(!canRemove)
                    throw new IllegalStateException("An element can not currently be removed");
                else
                {
                    canRemove = false;
                    removeHeapNode(pos+1);
                }
            }
        };
    }

    @Override
    public int size()
    {
        return size;
    }

    @Override
    public boolean offer(Integer e)
    {
        if(e == null)
            return false;
        return offer((int)e);
    }
    
    public boolean offer(int e)
    {
        int i = size++;
        if(heap.length < size)
            heap = Arrays.copyOf(heap, heap.length*2);
        heap[i] = e;
        if(fastValueRemove == Mode.HASH)
            valueIndexMap.put(e, i);
        else if (fastValueRemove == Mode.BOUNDED )
            if(e >= 0)
                indexArrayStore(e, i);
            else
            {
                heap[i] = 0;
                size--;
                return false;
            }
        
        heapifyUp(i);
        return true;
    }

    /**
     * Sets the given index to use the specific value
     * @param e the value to store the index of
     * @param i the index of the value
     */
    private void indexArrayStore(int e, int i)
    {
        if (valueIndexStore.length < e)
        {
            int oldLength = valueIndexStore.length;
            valueIndexStore = Arrays.copyOf(valueIndexStore, e + 2);
            Arrays.fill(valueIndexStore, oldLength, valueIndexStore.length, -1);
        }
        valueIndexStore[e] = i;
    }
    @SuppressWarnings("unused")
    private int getRightMostChild(int i)
    {
        int rightMostChild = i;
        //Get the right most child in this tree
        while(rightChild(rightMostChild) < size)
            rightMostChild = rightChild(rightMostChild);
        //Could be at a level with a left but no right
        while(leftChild(rightMostChild) < size)
            rightMostChild = leftChild(rightMostChild);
        
        return rightMostChild;
    }
    
    private int cmp(int i, int j)
    {
        return comparator.compare(heap[i], heap[j]);
    }

    private void heapDown(int i)
    {
        int iL = leftChild(i);
        int iR = rightChild(i);
        //While we have two children, make sure we are smaller
        while (childIsSmallerAndValid(i, iL) || childIsSmallerAndValid(i, iR))
        {
            //we are larger then one of ours children, so swap with the smallest of the two
            if ( iR < size && cmp(iL, iR) > 0 )//Right is the smallest
            {
                swapHeapValues(i, iR);
                i = iR;
            }
            else//Left is smallest or lef tis only option
            {
                swapHeapValues(i, iL);
                i = iL;
            }

            iL = leftChild(i);
            iR = rightChild(i);
        }
    }

    /**
     * Heapify up from the given index in the heap and make sure everything is 
     * correct. Stops when the child value is in correct order with its parent. 
     * @param i the index in the heap to start checking from. 
     */
    private void heapifyUp(int i)
    {
        int iP = parent(i);
        while(i != 0 && cmp(i, iP) < 0)//Should not be greater then our parent
        {
            swapHeapValues(iP, i);
            i = iP;
            iP = parent(i);
        }
    }

    /**
     * Swaps the values stored in the heap for the given indices
     * @param i the first index to be swapped 
     * @param j  the second index to be swapped
     */
    private void swapHeapValues(int i, int j)
    {
        if(fastValueRemove == Mode.HASH)
        {
            valueIndexMap.put(heap[i], j);
            valueIndexMap.put(heap[j], i);
        }
        else if(fastValueRemove == Mode.BOUNDED)
        {
            //Already in the array, so just need to set
            valueIndexStore[heap[i]] = j;
            valueIndexStore[heap[j]] = i;
        }
        int tmp = heap[i];
        heap[i] = heap[j];
        heap[j] = tmp;
    }

    private int parent(int i)
    {
        return (i-1)/2;
    }
    
    private int leftChild(int i)
    {
        return 2*i+1;
    }

    /**
     * Removes the node specified from the heap
     * @param i the valid heap node index to remove from the heap
     * @return the value that was stored in the heap node
     */
    protected int removeHeapNode(int i)
    {
        int val = heap[i];
        int rightMost = --size;
        heap[i] = heap[rightMost];
        heap[rightMost] = 0;
        if(fastValueRemove == Mode.HASH)
        {
            valueIndexMap.remove(val);
            if(size != 0)
                valueIndexMap.put(heap[i], i);
        }
        else if(fastValueRemove == Mode.BOUNDED)
        {
            valueIndexStore[val] = -1;
        }
        heapDown(i);
        return val;
    }
    
    private int rightChild(int i)
    {
        return 2*i+2;
    }

    private boolean childIsSmallerAndValid(int i, int child)
    {
        return child < size && cmp(i, child) > 0;
    }
    
    @Override
    public Integer poll()
    {
        if(isEmpty())
            return null;
        return removeHeapNode(0);
    }

    @Override
    public Integer peek()
    {
        if(isEmpty())
            return null;
        return heap[0];
    }

    @Override
    public boolean contains(Object o)
    {
        if(fastValueRemove == Mode.HASH)
            return valueIndexMap.containsKey(o);
        else if(fastValueRemove == Mode.BOUNDED)
        {
            if( o instanceof Integer)
            {
                int val = ((Integer) o).intValue();
                return val >= 0 && valueIndexStore[val] >= 0;
            }
            return false;
        }
        return super.contains(o);
    }

    @Override
    public void clear()
    {
        Arrays.fill(heap, 0, size, 0);
        size = 0;
        if(fastValueRemove == Mode.HASH)
            valueIndexMap.clear();
        else if(fastValueRemove == Mode.BOUNDED)
            Arrays.fill(valueIndexStore, -1);
    }

    @Override
    public boolean remove(Object o)
    {
        if(fastValueRemove == Mode.HASH)
        {
            Integer index = valueIndexMap.get(o);
            if(index == null)
                return false;
            removeHeapNode(index);
            return true;
        }
        else if(fastValueRemove == Mode.BOUNDED)
        {
            if(o instanceof Integer)
            {
                int val = ((Integer) o).intValue();
                if(val <0 || val >= valueIndexStore.length)
                    return false;
                int index = valueIndexStore[val];
                if(index == -1)
                    return false;
                removeHeapNode(index);
                return true;
            }
            return false;
        }
        return super.remove(o);
    }
    
}
