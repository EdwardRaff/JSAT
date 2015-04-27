
package jsat.classifiers.bayesian.graphicalmodel;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Provides a class representing an undirected graph. Mutations to the graph should be done 
 * exclusively through the methods provided by the class. Alterations should not be done to
 * the sets returned by any method. 
 * 
 * @author Edward Raff
 */
public class DirectedGraph<N> implements Cloneable
{    
    private static class Pair<A, B>
    {
        A incoming;
        B outgoing;

        public Pair(A first, B second)
        {
            this.incoming = first;
            this.outgoing = second;
        }

        public A getIncoming()
        {
            return incoming;
        }

        public B getOutgoing()
        {
            return outgoing;
        }
        @SuppressWarnings("unused")
        public void setIncoming(A first)
        {
            this.incoming = first;
        }
        @SuppressWarnings("unused")
        public void setOutgoing(B outgoing)
        {
            this.outgoing = outgoing;
        }

        @Override
        public boolean equals(Object obj)
        {
            if(obj == null || !(obj instanceof Pair))
                return false;
            Pair other = (Pair) obj;
            return this.incoming.equals(other.incoming) && this.outgoing.equals(other.outgoing);
        }

        @Override
        public int hashCode()
        {
            int hash = 7;
            hash = 79 * hash + (this.incoming != null ? this.incoming.hashCode() : 0);
            hash = 79 * hash + (this.outgoing != null ? this.outgoing.hashCode() : 0);
            return hash;
        }
    }
    
    
    /**
     * Represents the nodes and all the edges. Each node N, is mapped to its 
     * paired adjacency lists. 
     * <br><br>
     * The first list contains all the nodes that 
     * point to N, and the second list contains all the nodes that N 
     * points to. 
     */
    private Map<N, Pair<HashSet<N>, HashSet<N>>> nodes;

    public DirectedGraph()
    {
        nodes = new HashMap<N, Pair<HashSet<N>, HashSet<N>>>();
    }
   
    /**
     * Returns the set of all nodes currently in the graph
     * @return the set of all nodes in the graph
     */
    public Set<N> getNodes()
    {
        return nodes.keySet();
    }
    
    /**
     * Adds all the objects in <tt>c</tt> as nodes in the graph
     * @param c a collection of nodes to add
     */
    public void addNodes(Collection<? extends N> c)
    {
        for(N n : c)
            addNode(n);
    }
    
    /**
     * Adds a new node to the graph
     * @param node the object to make a node
     */
    public void addNode(N node)
    {
        if(!nodes.containsKey(node))
            nodes.put(node, new Pair<HashSet<N>, HashSet<N>>(new HashSet<N>(), new HashSet<N>()));
    }
    
    /**
     * Returns the set of all parents of the requested node, or null if the node does not exist in the graph 
     * @param n the node to obtain the parents of
     * @return the set of parents, or null if the node is not in the graph
     */
    public Set<N> getParents(N n)
    {
        Pair<HashSet<N>, HashSet<N>> p = nodes.get(n);
        
        if(p == null)
            return null;
        
        return p.getIncoming();
    }
    
    /**
     * Returns the set of all children of the requested node, or null if the node does not exist in the graph. 
     * @param n the node to obtain the children of
     * @return the set of parents, or null if the node is not in the graph
     */
    public Set<N> getChildren(N n)
    {
        Pair<HashSet<N>, HashSet<N>> p = nodes.get(n);
        
        if(p == null)
            return null;
        
        return p.getOutgoing();
    }
    
    /**
     * Removes the specified node from the graph. If the node was not in the graph, not change occurs
     * @param node the node to remove from the graph
     */
    public void removeNode(N node)
    {
        Pair<HashSet<N>, HashSet<N>> p = nodes.remove(node);
        if(p == null)
            return;
        //Outgoing edges we can ignore removint he node drops them. We need to avoid dangling incoming edges to this node we have removed
        HashSet<N> incomingNodes = p.getIncoming();
        for(N incomingNode : incomingNodes)
            nodes.get(incomingNode).getOutgoing().remove(node);
    }
    
    /**
     * Returns true if both <tt>a</tt> and <tt>b</tt> are nodes in the graph
     * @param a the first value to check for
     * @param b the second value to check for
     * @return true if both <tt>a</tt> and <tt>b</tt> are in the graph, false otherwise
     */
    private boolean containsBoth(N a, N b)
    {
        return nodes.containsKey(a) && nodes.containsKey(b);
    }
    
    /**
     * Adds a directed edge into the network from <tt>a</tt> to <tt>b</tt>. 
     * If <tt>a</tt> and <tt>b</tt> are not nodes in the graph, nothing occurs. 
     * @param a the parent node
     * @param b the child node
     */
    public void addEdge(N a, N b)
    {
        if( !containsBoth(a, b) )
            return;//Cant add nodes to things that doing exist
        nodes.get(a).getOutgoing().add(b);
        nodes.get(b).getIncoming().add(a);
    }
    
    /**
     * Removes a directed edge from the network connecting <tt>a</tt> to <tt>b</tt>. 
     * If <tt>a</tt> and <tt>b</tt> are not nodes in the graph, nothing occurs. 
     * @param a the parent node
     * @param b the child node
     */
    public void removeEdge(N a, N b)
    {
        if(!containsBoth(a, b))
            return;
        nodes.get(a).getOutgoing().remove(b);
        nodes.get(b).getIncoming().remove(a);
    }
    
    /**
     * Returns <tt>true</tt> if <tt>a</tt> is a node in the graph, or <tt>false</tt> otherwise. 
     * @param a the node in question
     * @return <tt>true</tt> if the node exists, <tt>false</tt> otherwise
     */
    public boolean containsNode(N a)
    {
        return nodes.containsKey(a);
    }

    @Override
    protected DirectedGraph<N> clone() 
    {
        DirectedGraph<N> clone = new DirectedGraph<N>();
        
        clone.addNodes(this.nodes.keySet());
        for(N key : nodes.keySet())
        {
            Pair<HashSet<N>, HashSet<N>> p = nodes.get(key);
            for(N n : p.getIncoming())
                clone.nodes.get(key).getIncoming().add(n);
            for(N n : p.getOutgoing())
                clone.nodes.get(key).getOutgoing().add(n);
            
        }
        
        return clone;
    }

    
    
}


