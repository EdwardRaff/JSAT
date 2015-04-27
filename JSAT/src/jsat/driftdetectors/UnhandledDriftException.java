package jsat.driftdetectors;

/**
 * This exception is thrown when a drift detector receives new data even through 
 * the drift was not handled. 
 * 
 * @author Edward Raff
 */
public class UnhandledDriftException extends RuntimeException
{


	private static final long serialVersionUID = -5781626293819651067L;

	public UnhandledDriftException()
    {
        super();
    }

    public UnhandledDriftException(String message)
    {
        super(message);
    }
}
