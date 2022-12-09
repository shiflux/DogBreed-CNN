
export default function NavBar () {
    return (
        <nav className="navbar navbar-expand-sm navbar-dark d-flex justify-content-between px-5 bg-dark">
            <a className="navbar-brand" href="#">
                <img src='/logo.png' width={50} /> 
                <span className='mx-3'>Dog Breed Idenfitier</span>
                
            </a>
            <button className="navbar-toggler d-lg-none" type="button" data-bs-toggle="collapse" data-bs-target="#collapsibleNavId" aria-controls="collapsibleNavId"
                aria-expanded="false" aria-label="Toggle navigation"></button>
            <div className="collapse navbar-collapse w-auto">
                <ul className="navbar-nav mt-0 mt-lg-0 ">
                    <li className="nav-item">
                        <a className="nav-link active" href="#" aria-current="page">Home <span className="visually-hidden">(current)</span></a>
                    </li>
                    <li>
                        <a className="nav-link" href="https://github.com/shiflux/" target='_blank'>Github</a>
                    </li>
                </ul>
            </div>
        </nav>
        )
}