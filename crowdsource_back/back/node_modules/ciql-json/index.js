const fs = require('fs')
const { resolve } = require('path')
const path = require('path')


module.exports = {
    data: {},
    file: '',
    init() {
        this.data = {}
        this.file = ''
    },
    create(object) {
        try {
            if (object instanceof Object) {
                this.init()
                this.data = object
                return this
            } else {
                console.error("Error (Function create) : accepts only Object type as arg");
            }
        } catch (error) {
            console.error("Error (Function create) : ", error.message);
        }
    },
    extract(schema) {
        try {
            eval(`
            const ${schema}=this.data; 
            this.data = ${schema}
            `)
            return this
        } catch (error) {
            console.error("Error (Function extract) : ", error.message);
        }
    },
    open(input) {
        try {
            if (fs.existsSync(input)) {
                let file = path.resolve(input)
                if (path.extname(file) === '.json') {
                    this.init()
                    this.data = require(file)
                    this.file = file
                    return this
                } else {
                    console.error("Error (Function open) : ", input, "It's not a json file");
                }
            } else {
                console.error("Error (Function open) : ", input, "Not exist");
            }
        } catch (error) {
            console.error("Error (Function open) : ", error.message)
        }

    },
    set(key, value) {
        try {
            eval(`this.data.${key} = value`)
        } catch (error) {
            try {
                this.data[key] = value
            } catch (error) {
                console.error("Error (Function set) : ", error.message);
            }
        } finally {
            return this
        }

    },
    remove(key) {
        try {
            delete this.data[key]
        } catch (error) {
            console.error("Error (Function remove) : ", error.message);
        } finally {
            return this
        }
    },
    pushTo(key, ...values) {
        try {
            eval(`this.data.${key}.push(...values)`)
        } catch (error) {
            try {
                this.data[key].push(...values)
            } catch (error) {
                console.error("Error (Function pushTo) : ", error.message);
            }
        } finally {
            return this
        }

    },
    popTo(key) {
        try {
            eval(`this.data.${key}.pop()`)
        } catch (error) {
            try {
                this.data[key].pop()
            } catch (error) {
                console.error("Error (Function popTo) : ", error.message);
            }
        } finally {
            return this
        }

    },
    save(output) {
        try {
            fs.writeFileSync(resolve(output || this.file), JSON.stringify(this.data, null, 4))
        } catch (error) {
            console.error("Error (Function save): ", error.message);
        } finally {
            this.init()
        }
    },
    getData() {
        try {
            return this.data
        } catch (error) {
            console.error("Error (Function getData) : ", error.message);
        } finally {
            this.init()
        }
    },
    getValues() {
        try {
            return Object.values(this.data)
        } catch (error) {
            console.error("Error (Function getValues) : ", error.message);
        } finally {
            this.init()
        }
    },
    getKeys() {
        try {
            return Object.keys(this.data)
        } catch (error) {
            console.error("Error (Function getKeys) : ", error.message);
        } finally {
            this.init()
        }
    }

}