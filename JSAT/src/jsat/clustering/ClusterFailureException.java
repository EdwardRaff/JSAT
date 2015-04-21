package jsat.clustering;

/**
 * A ClusterFailureException is thrown when a clustering method is unable to 
 * perform its clustering for some reason. 
 * 
 * @author Edward Raff
 */
public class ClusterFailureException extends RuntimeException
{


	private static final long serialVersionUID = -8084320940762402095L;

	public ClusterFailureException()
    {
    }

    public ClusterFailureException(String string)
    {
        super(string);
    }

    public ClusterFailureException(Throwable thrwbl)
    {
        super(thrwbl);
    }

    public ClusterFailureException(String string, Throwable thrwbl)
    {
        super(string, thrwbl);
    }
    
}
