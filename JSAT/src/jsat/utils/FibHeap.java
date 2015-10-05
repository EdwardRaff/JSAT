/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.utils;

import java.util.*;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 * @param <T>
 */
public class FibHeap<T>
{
    /**
     * We access a given Fibonacci heap H by a pointer H:min to the root of a tree containing the minimum key. When a Fibonacci heap H is empty, H:min is NIL.
     */
    FibNode<T> min;
    /**
     * We rely on one other attribute for a Fibonacci heap H: H:n, the number of
     * nodes currently in H
     */
    int n;
    
    List<FibNode<T>> H;
    

    public FibHeap()
    {
        /**
         * To make an empty Fibonacci heap, the MAKE-FIB-HEAP procedure
         * allocates and returns the Fibonacci heap object H, where H:n D 0 and
         * H:min D NIL; there are no trees in H.
         */
        this.min = null;
        this.n = 0;
    }
    
    public int size()
    {
        return n;
    }
    
    public FibNode<T> insert(T value, double weight)
    {
        //FIB-HEAP-INSERT
        @SuppressWarnings("unchecked")
        FibNode<T> x = new FibNode(value, weight);
        x.p = null;
        x.child = null;
        
        if(min == null)
        {
            //6| create a root list for H containing just x
//            H = new ArrayList<FibNode<T>>();
//            H.add(x);
            //7| H.min D x
            min = x;
        }
        else
        {
            //8| insert x into H ’s root list
            min = merge(min, x);//9 & 10 handled implicitly by the behavior of merge method
        }
        //11| H.n = H.n+1
        this.n++;
        
        return x;
    }
    
    public static <T> FibHeap<T> union(FibHeap<T> A, FibHeap<T> B)
    {
        FibHeap<T> H = new FibHeap<T>();

        H.min = merge(A.min, B.min);
        H.n = A.n + B.n;
        
        return H;
    }
    
    private void consolidate()
    {
        ArrayList<FibNode<T>> A = new ArrayList<FibNode<T>>();
        ArrayList<FibNode<T>> rootListCopy = new ArrayList<FibNode<T>>();
        
        FibNode<T> startingNode = min;
        FibNode<T> runner = startingNode;
        do
        {
            rootListCopy.add(runner);
            //go to the next item in the list
            runner = runner.right;
        }
        while(runner != startingNode);//Yes, intentionally comparing objects. Circular link list. So once we reach the starting object we are done
        
        for(FibNode<T> w : rootListCopy)
            delink(w);//we will fix this later by re-merging at the end
        
        //4| for each node w in the root list of H
        for(FibNode<T> w : rootListCopy)
        {
            FibNode<T> x = w;//5
            int d = x.degree;//6
            
            //7| while A[d] != NIL
            while(A.size() <= d)
                A.add(null);
            while(A.get(d) != null)
            {
                //8| y = A[d]  // another node with the same degree as x
                FibNode<T> y = A.get(d);
                if(x.key > y.key)
                {
                    FibNode<T> tmp = y;
                    y = x;
                    x = tmp;
                }
                //11| FIB-HEAP-LINK(H,y,x)
//                if(startingNode == y )//if y was our starting node, we need to change the start to avoid a loop
//                    startingNode = y.right;//otherwise looping through root list will never reach the original starting node
//                if(y == w)//move w left so that when we move right later we get to the correct position
//                    w = w.left;//otherwise w will start looping through the child list isntead of the root list
                link(y, x);
                //12| A[d]=NIL
                A.set(d, null);
                //13| d=d+1
                d++;
                while(A.size() <= d)
                    A.add(null);
            }
            //14| A[d] = x
            A.set(d, x);
        }
        
        //15| H.min = NIL
        min = null;
        for(FibNode<T> x : A)
            min = merge(min, x);
        
    }
    
    private void link(FibNode<T> y, FibNode<T> x)
    {
        //1| remove y from the root list of H
        delink(y);
        //2| make y a child of x, incrementing x.degree 
        x.addChild(y);
        x.degree++;//adding y as a child
        x.degree += y.degree;//and all of y's children
        //3| y.mark = FALSE
        y.mark = false;
    }
    
    public double getMinKey()
    {
        return min.key;
    }
    
    public T getMinValue()
    {
        return min.value;
    }
    
    public FibNode<T> peekMin()
    {
        return min;
    }
    
    public FibNode<T> removeMin()
    {
        //FIB-HEAP-EXTRACT-MIN
        //1| z = H.min
        FibNode<T> z = min;
        if(z != null)
        {   
            min = delink(z);
            
            //3| for each child x of z
            if (z.child != null)
            {
                final FibNode<T> startingNode = z.child;
                FibNode<T> x = startingNode;
                do
                {
                    x.p = null;//set parrent to null, we will add to the root list later
                    //go to the next item in the list
                    x = x.right;
                }
                while (x != startingNode);//Yes, intentionally comparing objects. Circular link list. So once we reach the starting object we are done
                
                //now all the children are added to the root list in one call
                min = merge(min, z.child);
                z.child = null;//z has no more children
                z.degree = 0;
            }
            else if (n == 1)//z had no children and was the only item in the whole heap
            {
                //so just set min to null
                min = null;
            }
           
            if(min != null)
                consolidate();//this will set min correctly no matter what
            n--;
        }

        return z;
    }
    
    public void decreaseKey(FibNode<T> x, double k)
    {
        if(k > x.key)
            throw new RuntimeException("new key is greater than current key");
        x.key = k;
        FibNode<T> y = x.p;
        if(y != null && x.key < y.key)
        {
            cut(x, y);
            cascadingCut(y);
        }
        if(x.key < min.key)
            min = x;
    }
    
    /**
     * 
     * @param x the child
     * @param y the parent
     */
    private void cut(FibNode<T> x, FibNode<T> y)
    {
        //1| remove x from the child list of y, decrementing y.degree
        if(y.child == x)//if we removed but x was the child pointer, we would get messed up
            y.child = delink(x);
        else
            delink(x);
        y.degree--;//removal of 'x' from y
        y.degree-= x.degree;//removal of everyone x owned from y
        //2| add x to the root list of H
        min = merge(min, x);
        //3| x.p = NIL
        x.p = null;
        //4| x.mark = FALSE
        x.mark = false;
    }
    
    private void cascadingCut(FibNode<T> y)
    {
        //1.
        FibNode<T> z = y.p;
        if(z != null)//2.
        {
            if(y.mark == false)//3.
                y.mark = true;//4.
            else
            {
                cut(y, z);
                cascadingCut(z);
            }
        }
    }
   
    
    public static class FibNode<T>
    {
        
        T value;
        double key;
        
        /**
         * We store the number of children in the child list of node x in x.degree
         */
        int degree;
        
        /**
         * The boolean-valued attribute x:mark indicates whether node x has lost
         * a child since the last time x was made the child of another node.
         * Newly created nodes are unmarked, and a node x becomes unmarked
         * whenever it is made the child of another node.
         */
        boolean mark = false;
        
        /**
         * a pointer to its parent
         */
        FibNode<T> p;
        /**
         * a pointer to any one of its children
         */
        FibNode<T> child;
        
        FibNode<T> left, right;

        public FibNode(T value, double key)
        {
            this.degree = 0;
            this.value = value;
            this.key = key;
            this.left = this.right = this;//yes, this is intentional. We make a linked list of 1 item, ourselves. 
        }

        public T getValue()
        {
            return value;
        }
        
        public double getPriority()
        {
            return key;
        }
        
        /**
         * Adds the given node directly to the children list of this node. THe
         * {@link #p} value for {@code x} will be set automatically. The
         * {@link #degree} of this node will not be adjusted, and must be
         * incremented correctly by the caller<br>
         *
         * @param x the node to add as a child of this node. Should be a
         * singular item
         */
        public void addChild(FibNode<T> x)
        {
            if(this.child == null)
                this.child = x;
            else
                this.child = merge(this.child, x);
            x.p = this;
        }

        @Override
        public String toString()
        {
            return value + "," + key;
        }
        
    }
    
    /**
     * Disconnects the given node from the list it is currently in.
     *
     * @param <T>
     * @param a the node to disconnect from its current list
     * @return a node in the list that 'a' was originally apart of. Returns
     * {@code null} if a was its own list (ie: a list of size 1, or no "list").
     */
    private static <T> FibNode<T> delink(FibNode<T> a)
    {
        if(a.left == a)//a is on its own, nothing to return
            return null;
        //else, a is in a list
        
        FibNode<T> a_left_orig = a.left;//link to the rest of the list that we can return
        
        a.left.right = a.right;
        a.right.left = a_left_orig;
        //a no longer is in its original list, return link
        
        //fix a's links to point to itself not that it has been de-linked from everyone else
        a.left = a.right = a;
        
        return a_left_orig;
    }
    
    /**
     * Merges the two lists given by the two nodes. Works if either node represents a list of size 1 / is on its own. The node with the smaller value will be returned. 
     * @param <T>
     * @param a
     * @param b
     * @return the node with the smallest key value
     */
    private static <T> FibNode<T> merge(FibNode<T> a, FibNode<T> b)
    {
        if(a == b)//handles a and b == null (returns null) or a and b are the same object (return the obj, don't make weird cycles
            return a;
        if(a == null)
            return b;
        if(b == null)
            return a;
        
        //else, two different non-null nodes, make a the smaller priority one so we always return smallest priority
        if(a.key > b.key)
        {
            FibNode<T> tmp = a;
            a = b;
            b = tmp;
        }
  
        //linked list insert. Indertion used since links can point to themselves if list is empty
        FibNode<T> a_right_orig = a.right; 
        a.right = b.right;
        a.right.left = a;
        b.right = a_right_orig;
        b.right.left = b;
        
        return a;
    }
            
}
