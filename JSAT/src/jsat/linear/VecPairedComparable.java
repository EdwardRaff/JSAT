package jsat.linear;

/**
 * Utility class for using {@link VecPaired} when the paired value is comparable
 * , and the vectors need to be sorted based on their paired value. This class 
 * performs exactly the same, and its only modification is that it is comparable
 * based on the paired object type. 
 * 
 * @author Edward Raff
 */
public class VecPairedComparable<V extends Vec, P extends Comparable<P>> extends VecPaired<V, P> implements Comparable<VecPairedComparable<V, P>>
{


	private static final long serialVersionUID = -7061543870162459467L;

	public VecPairedComparable(V v, P p)
    {
        super(v, p);
    }

    @Override
    public int compareTo(VecPairedComparable<V, P> o)
    {
        return this.getPair().compareTo(o.getPair());
    }
}
