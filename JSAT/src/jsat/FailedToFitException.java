
package jsat;

/**
 *
 * @author Edward Raff
 */
public class FailedToFitException extends RuntimeException
{
    private Exception faultException;

    public FailedToFitException(Exception faultException, String message)
    {
        super(message);
        this.faultException = faultException;
    }

    public FailedToFitException(Exception faultException, Throwable cause)
    {
        super(cause);
        this.faultException = faultException;
    }

    public FailedToFitException(Exception faultException, String message, Throwable cause)
    {
        super(message, cause);
        this.faultException = faultException;
    }

    public FailedToFitException(Exception faultException)
    {
        super(faultException.getMessage());
        this.faultException = faultException;
    }

    /**
     * Returns the exception that caused the issue. 
     * @return the exception that caused the issue. 
     */
    public Exception getFaultException()
    {
        return faultException;
    }
    
    
}
