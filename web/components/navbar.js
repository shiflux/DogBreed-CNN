import {useRouter} from 'next/router';

export default function NavBar () {
    const router = useRouter()
    return (
        <nav className="navbar navbar-expand-sm navbar-dark d-flex justify-content-between px-5 bg-dark mb-5">
            <a className="navbar-brand" href="/">
                <img src='/logo.png' width={50} /> 
                <span className='mx-3 d-none d-md-inline-block'>Dog Breed Finder</span>
                
            </a>
            <div className=" w-auto">
                <ul className="navbar-nav mt-0 mt-lg-0 flex-row">
                    
                    <li className="nav-item mx-3 md-mx-4 w-fit">
                        <a className={'nav-link' + (router.pathname==='/dogs' ? ' active' : '')} href="dogs" aria-current="page">List of Dogs<span className="visually-hidden">(current)</span></a>
                    </li>
                    <li className='w-fit'>
                        <a className="nav-link w-fit" href="https://github.com/shiflux/" target='_blank'><img src='/github.png' className='d-inline-block bg-white rounded' width={45} /></a>
                    </li>
                </ul>
            </div>
        </nav>
        )
}